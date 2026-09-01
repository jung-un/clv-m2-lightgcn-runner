"""Checkpoint-only diagnostic for the CLV category-transition M3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m3_clv_conditioned_category_transition_graph import (
    ARM_ACTUAL,
    ARM_SHUFFLE,
    build_clv_conditioned_category_transition_graph,
)
from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_m3_clv_conditioned_category_transition as source_runner
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-category-transition-failure-diagnostic-v1"


@dataclass(frozen=True)
class M3CategoryTransitionDiagnosticConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    rank_limit: int = 50
    score_batch_size: int = 64
    source_result_id: str = "fdf845b331c2"
    source_out_dir: str = ""
    diagnostic_out_dir: str = ""


def configure_m3_category_transition_diagnostic(
    **overrides,
) -> M3CategoryTransitionDiagnosticConfig:
    source = (
        f"{v3.default_out_dir('dunnhumby')}"
        "_m3_clv_category_transition_historical_screen_v1"
    )
    defaults = {
        "source_out_dir": source,
        "diagnostic_out_dir": f"{source}/failure_mechanism_diagnostic_v1",
    }
    cfg = M3CategoryTransitionDiagnosticConfig(**(defaults | overrides))
    if cfg.dataset != "dunnhumby" or cfg.seed != 42 or cfg.rank_limit != 50:
        raise ValueError("진단은 Dunnhumby seed 42, Top-50으로 고정합니다")
    if cfg.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    if not cfg.source_result_id or not cfg.source_out_dir or not cfg.diagnostic_out_dir:
        raise ValueError("source result and output directories are required")
    return cfg


def preflight_summary(cfg: M3CategoryTransitionDiagnosticConfig) -> dict:
    cfg = configure_m3_category_transition_diagnostic(**cfg.__dict__)
    return {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc checkpoint diagnostic",
        "source_models": [source_runner.ACTUAL_ID, source_runner.SHUFFLE_ID],
        "source_split": "DAY 1--683 train; DAY 684--690 evaluation",
        "training": False,
        "checkpoint_selection": False,
        "final_test_constructed": False,
        "holdout_constructed": False,
        "rank_limit": cfg.rank_limit,
        "source_result_id": cfg.source_result_id,
        "questions": [
            "are actual and shuffled CLV relation rows materially different?",
            "does the auxiliary relation produce score variation near Top-50?",
            "does that variation change Top-10/20/50 recommendations by CLV quintile?",
        ],
        "interpretation": (
            "hypothesis generation only; no retraining, model selection, "
            "significance, or generalization claim"
        ),
        "source_out_dir": cfg.source_out_dir,
        "diagnostic_out_dir": cfg.diagnostic_out_dir,
    }


@torch.no_grad()
def score_pairs_from_parts(
    base_user: torch.Tensor,
    base_item: torch.Tensor,
    message: torch.Tensor,
    *,
    users: np.ndarray,
    items: np.ndarray,
    gamma: float,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    if users.shape != items.shape or users.ndim != 1:
        raise ValueError("users and items must be aligned vectors")
    if batch_size <= 0 or gamma <= 0:
        raise ValueError("batch_size and gamma must be positive")
    base_values = []
    auxiliary_values = []
    for start in range(0, len(users), batch_size):
        user_index = torch.as_tensor(
            users[start : start + batch_size],
            dtype=torch.long,
            device=base_user.device,
        )
        item_index = torch.as_tensor(
            items[start : start + batch_size],
            dtype=torch.long,
            device=base_item.device,
        )
        selected_item = base_item.index_select(0, item_index)
        base_values.append(
            (
                base_user.index_select(0, user_index) * selected_item
            ).sum(1).cpu().numpy()
        )
        auxiliary_values.append(
            (
                gamma
                * (message.index_select(0, user_index) * selected_item).sum(1)
            ).cpu().numpy()
        )
    return (
        np.concatenate(base_values).astype(np.float64),
        np.concatenate(auxiliary_values).astype(np.float64),
    )


def graph_similarity_summary(per_user: pd.DataFrame) -> pd.DataFrame:
    required = {
        "q_actual",
        "edge_jaccard",
        "total_variation_distance",
        "weight_cosine",
        "exact_relation_row",
    }
    missing = required - set(per_user.columns)
    if missing:
        raise ValueError(f"graph similarity rows miss {sorted(missing)}")
    frame = per_user.copy()
    frame["clv_group"] = _clv_quintile(frame["q_actual"].to_numpy())
    rows = []
    for group in ("전체", "Q1", "Q2", "Q3", "Q4", "Q5"):
        selected = frame if group == "전체" else frame.loc[frame["clv_group"].eq(group)]
        if selected.empty:
            continue
        rows.append(
            {
                "clv_group": group,
                "n_users": int(len(selected)),
                "mean_edge_jaccard": float(selected["edge_jaccard"].mean()),
                "median_edge_jaccard": float(selected["edge_jaccard"].median()),
                "mean_total_variation_distance": float(
                    selected["total_variation_distance"].mean()
                ),
                "median_total_variation_distance": float(
                    selected["total_variation_distance"].median()
                ),
                "mean_weight_cosine": float(selected["weight_cosine"].mean()),
                "exact_relation_user_share": float(
                    selected["exact_relation_row"].astype(float).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def clv_assignment_correlation(per_user: pd.DataFrame) -> pd.DataFrame:
    required = {"q_actual", "q_shuffle", "shuffle_stratum"}
    missing = required - set(per_user.columns)
    if missing:
        raise ValueError(f"CLV assignment rows miss {sorted(missing)}")
    groups = [("전체", per_user)]
    groups.extend(
        (f"degree_stratum_{int(stratum)}", selected)
        for stratum, selected in per_user.groupby("shuffle_stratum", sort=True)
    )
    rows = []
    for scope, selected in groups:
        correlation = selected["q_actual"].corr(
            selected["q_shuffle"], method="spearman"
        )
        rows.append(
            {
                "scope": scope,
                "n_users": int(len(selected)),
                "q_actual_shuffle_spearman": float(correlation),
                "mean_absolute_q_displacement": float(
                    np.abs(selected["q_actual"] - selected["q_shuffle"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def score_component_summary(
    base_scores: np.ndarray,
    auxiliary_scores: np.ndarray,
    *,
    q_actual: np.ndarray,
    model_id: str,
    candidate_role: str,
) -> pd.DataFrame:
    base_scores = np.asarray(base_scores, dtype=np.float64)
    auxiliary_scores = np.asarray(auxiliary_scores, dtype=np.float64)
    q_actual = np.asarray(q_actual, dtype=np.float64)
    if base_scores.shape != auxiliary_scores.shape or base_scores.ndim != 2:
        raise ValueError("base and auxiliary score arrays must align in two dimensions")
    if len(q_actual) != len(base_scores):
        raise ValueError("CLV percentiles must align with score rows")
    if not model_id or not candidate_role:
        raise ValueError("model_id and candidate_role are required")
    if not np.isfinite(base_scores).all() or not np.isfinite(auxiliary_scores).all():
        raise ValueError("score components must be finite")
    quintile = _clv_quintile(q_actual)
    rows = []
    for group in ("전체", "Q1", "Q2", "Q3", "Q4", "Q5"):
        mask = (
            np.ones(len(base_scores), dtype=bool)
            if group == "전체"
            else quintile == group
        )
        if not mask.any():
            continue
        base = base_scores[mask].reshape(-1)
        auxiliary = auxiliary_scores[mask].reshape(-1)
        base_std = float(base.std())
        auxiliary_std = float(auxiliary.std())
        rows.append(
            {
                "model_id": model_id,
                "candidate_role": candidate_role,
                "clv_group": group,
                "n_users": int(mask.sum()),
                "candidate_pair_count": int(base.size),
                "base_score_mean": float(base.mean()),
                "base_score_std": base_std,
                "auxiliary_score_mean": float(auxiliary.mean()),
                "auxiliary_score_std": auxiliary_std,
                "mean_abs_auxiliary_score": float(np.abs(auxiliary).mean()),
                "auxiliary_to_base_std_ratio": (
                    auxiliary_std / base_std if base_std > 0 else np.nan
                ),
                "full_score_std": float((base + auxiliary).std()),
            }
        )
    return pd.DataFrame(rows)


def score_component_long_summary(
    *,
    base_scores: np.ndarray,
    auxiliary_scores: np.ndarray,
    score_users: np.ndarray,
    q_by_user: np.ndarray,
    model_id: str,
    candidate_role: str,
) -> pd.DataFrame:
    base_scores = np.asarray(base_scores, dtype=np.float64)
    auxiliary_scores = np.asarray(auxiliary_scores, dtype=np.float64)
    score_users = np.asarray(score_users, dtype=np.int64)
    q_by_user = np.asarray(q_by_user, dtype=np.float64)
    if not (
        base_scores.ndim == auxiliary_scores.ndim == score_users.ndim == 1
        and len(base_scores) == len(auxiliary_scores) == len(score_users)
    ):
        raise ValueError("score components and score_users must be aligned vectors")
    if not len(base_scores):
        raise ValueError("at least one scored pair is required")
    if score_users.min(initial=0) < 0 or score_users.max(initial=-1) >= len(q_by_user):
        raise ValueError("score user index is outside q_by_user")
    if not np.isfinite(base_scores).all() or not np.isfinite(auxiliary_scores).all():
        raise ValueError("score components must be finite")
    user_quintile = _clv_quintile(q_by_user)
    pair_quintile = user_quintile[score_users]
    rows = []
    for group in ("전체", "Q1", "Q2", "Q3", "Q4", "Q5"):
        mask = (
            np.ones(len(base_scores), dtype=bool)
            if group == "전체"
            else pair_quintile == group
        )
        if not mask.any():
            continue
        base = base_scores[mask]
        auxiliary = auxiliary_scores[mask]
        base_std = float(base.std())
        auxiliary_std = float(auxiliary.std())
        rows.append(
            {
                "model_id": model_id,
                "candidate_role": candidate_role,
                "clv_group": group,
                "n_users": int(np.unique(score_users[mask]).size),
                "candidate_pair_count": int(mask.sum()),
                "base_score_mean": float(base.mean()),
                "base_score_std": base_std,
                "auxiliary_score_mean": float(auxiliary.mean()),
                "auxiliary_score_std": auxiliary_std,
                "mean_abs_auxiliary_score": float(np.abs(auxiliary).mean()),
                "auxiliary_to_base_std_ratio": (
                    auxiliary_std / base_std if base_std > 0 else np.nan
                ),
                "full_score_std": float((base + auxiliary).std()),
            }
        )
    return pd.DataFrame(rows)


def _clv_quintile(q_values: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    if not np.isfinite(q_values).all() or np.any((q_values < 0) | (q_values > 1)):
        raise ValueError("CLV percentiles must be finite and inside [0, 1]")
    number = np.minimum((q_values * 5).astype(np.int64) + 1, 5)
    return np.asarray([f"Q{value}" for value in number], dtype=object)


def recommendation_overlap_summary(
    reference: np.ndarray,
    model: np.ndarray,
    *,
    q_actual: np.ndarray,
    comparison: str,
    ks: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    reference = np.asarray(reference)
    model = np.asarray(model)
    q_actual = np.asarray(q_actual, dtype=np.float64)
    if reference.shape != model.shape or reference.ndim != 2:
        raise ValueError("recommendation arrays must be aligned two-dimensional arrays")
    if len(q_actual) != len(reference):
        raise ValueError("CLV percentiles must align with recommendation rows")
    if not comparison:
        raise ValueError("comparison name is required")
    if not ks or min(ks) <= 0 or max(ks) > reference.shape[1]:
        raise ValueError("every cutoff must fit inside the recommendation width")
    quintile = _clv_quintile(q_actual)
    rows = []
    for k in ks:
        left = reference[:, :k]
        right = model[:, :k]
        set_changed = np.empty(len(left), dtype=np.float64)
        order_changed = np.empty(len(left), dtype=np.float64)
        jaccard = np.empty(len(left), dtype=np.float64)
        for row, (left_items, right_items) in enumerate(
            zip(left, right, strict=True)
        ):
            left_set = set(map(int, left_items))
            right_set = set(map(int, right_items))
            set_changed[row] = float(left_set != right_set)
            order_changed[row] = float(not np.array_equal(left_items, right_items))
            jaccard[row] = len(left_set & right_set) / max(
                len(left_set | right_set), 1
            )
        for group in ("전체", "Q1", "Q2", "Q3", "Q4", "Q5"):
            mask = (
                np.ones(len(left), dtype=bool)
                if group == "전체"
                else quintile == group
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "comparison": comparison,
                    "clv_group": group,
                    "k": int(k),
                    "n_users": int(mask.sum()),
                    "set_changed_user_share": float(set_changed[mask].mean()),
                    "order_changed_user_share": float(order_changed[mask].mean()),
                    "mean_jaccard": float(jaccard[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def _sparse_rows(matrix: torch.Tensor) -> list[dict[int, float]]:
    if matrix.layout != torch.sparse_coo:
        raise ValueError("relation operator must be sparse COO")
    matrix = matrix.coalesce().cpu()
    rows = [dict() for _ in range(matrix.shape[0])]
    indices = matrix.indices().numpy()
    values = matrix.values().double().numpy()
    for row, column, value in zip(
        indices[0], indices[1], values, strict=True
    ):
        rows[int(row)][int(column)] = float(value)
    return rows


def sparse_row_similarity(
    actual: torch.Tensor,
    shuffled: torch.Tensor,
    *,
    q_actual: np.ndarray,
    q_shuffle: np.ndarray,
    strata: np.ndarray,
) -> pd.DataFrame:
    if actual.shape != shuffled.shape:
        raise ValueError("actual and shuffled operators must have the same shape")
    n_users = int(actual.shape[0])
    q_actual = np.asarray(q_actual, dtype=np.float64)
    q_shuffle = np.asarray(q_shuffle, dtype=np.float64)
    strata = np.asarray(strata)
    if any(len(values) != n_users for values in (q_actual, q_shuffle, strata)):
        raise ValueError("CLV assignments and strata must align with graph rows")
    actual_rows = _sparse_rows(actual)
    shuffled_rows = _sparse_rows(shuffled)
    records = []
    for user, (left, right) in enumerate(
        zip(actual_rows, shuffled_rows, strict=True)
    ):
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        common = left_set & right_set
        left_norm = float(np.sqrt(sum(value * value for value in left.values())))
        right_norm = float(np.sqrt(sum(value * value for value in right.values())))
        dot = float(sum(left[column] * right[column] for column in common))
        total_variation = 0.5 * sum(
            abs(left.get(column, 0.0) - right.get(column, 0.0))
            for column in union
        )
        cosine = (
            dot / (left_norm * right_norm)
            if left_norm > 0 and right_norm > 0
            else np.nan
        )
        records.append(
            {
                "user_idx": user,
                "q_actual": float(q_actual[user]),
                "q_shuffle": float(q_shuffle[user]),
                "shuffle_stratum": int(strata[user]),
                "actual_edge_count": len(left_set),
                "shuffle_edge_count": len(right_set),
                "common_edge_count": len(common),
                "edge_jaccard": len(common) / max(len(union), 1),
                "total_variation_distance": float(total_variation),
                "weight_cosine": float(cosine),
                "exact_relation_row": bool(left == right),
            }
        )
    return pd.DataFrame(records)


def _source_result(cfg: M3CategoryTransitionDiagnosticConfig) -> tuple[Path, dict]:
    path = Path(cfg.source_out_dir) / (
        f"m3_clv_category_transition_{cfg.source_result_id}.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("code_version") != source_runner.CODE_VERSION:
        raise RuntimeError("source result code_version does not match the M3 runner")
    if payload.get("config", {}).get("seed") != cfg.seed:
        raise RuntimeError("source result seed does not match the diagnostic")
    return path, payload


def _prepare_source(
    cfg: M3CategoryTransitionDiagnosticConfig,
    source_payload: dict,
) -> tuple[dict, source_runner.CLVCategoryTransitionConfig, str]:
    source_cfg = source_runner.validate_clv_category_transition_config(
        source_runner.CLVCategoryTransitionConfig(**source_payload["config"])
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
    graph = build_clv_conditioned_category_transition_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        data["n_cat"],
        kappa=source_cfg.kappa,
        min_support_users=source_cfg.min_support_users,
        log_lift_cap=source_cfg.log_lift_cap,
        shuffle_seed=source_cfg.shuffle_seed,
        shuffle_degree_bins=source_cfg.shuffle_degree_bins,
        cross_fit_folds=source_cfg.cross_fit_folds,
        max_target_categories=source_cfg.max_target_categories,
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
    cfg: M3CategoryTransitionDiagnosticConfig,
    prepared: dict,
    source_cfg: source_runner.CLVCategoryTransitionConfig,
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
        "model_id": model_id,
        "graph_arm": arm,
        "seed": cfg.seed,
        "input_hash": prepared["input_hash"],
    }
    mismatches = {
        key: {"expected": value, "actual": blob.get(key)}
        for key, value in expected.items()
        if blob.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint identity mismatch: {mismatches}")
    model = source_runner._build_model(prepared, source_cfg, arm)
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, record, checkpoint


@torch.no_grad()
def _model_views(model, prepared: dict, cfg: M3CategoryTransitionDiagnosticConfig):
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


def _score_pair_population(
    views: dict[str, dict],
    cache: v3.EvalCache,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    actual = views[ARM_ACTUAL]
    shuffled = views[ARM_SHUFFLE]
    users = actual["users"]
    union_users = []
    union_items = []
    for user, left, right in zip(
        users,
        actual["full_top50"],
        shuffled["full_top50"],
        strict=True,
    ):
        for item in sorted(set(map(int, left)) | set(map(int, right))):
            union_users.append(int(user))
            union_items.append(item)
    truth_users = []
    truth_items = []
    for user in users:
        for item in cache.gt[int(user)]:
            truth_users.append(int(user))
            truth_items.append(int(item))
    return {
        "actual_shuffle_top50_union": (
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
    for arm, model_id in (
        (ARM_ACTUAL, source_runner.ACTUAL_ID),
        (ARM_SHUFFLE, source_runner.SHUFFLE_ID),
    ):
        view = views[arm]
        for role, (users, items) in populations.items():
            base, auxiliary = score_pairs_from_parts(
                view["base_user"],
                view["base_item"],
                view["message"],
                users=users,
                items=items,
                gamma=view["gamma"],
            )
            tables.append(
                score_component_long_summary(
                    base_scores=base,
                    auxiliary_scores=auxiliary,
                    score_users=users,
                    q_by_user=q_actual,
                    model_id=model_id,
                    candidate_role=role,
                )
            )
    components = pd.concat(tables, ignore_index=True)
    value = components.pivot_table(
        index=["model_id", "clv_group"],
        columns="candidate_role",
        values=["auxiliary_score_mean", "mean_abs_auxiliary_score"],
        aggfunc="first",
    )
    value.columns = [f"{metric}__{role}" for metric, role in value.columns]
    contrast = value.reset_index()
    contrast["truth_minus_top50_auxiliary_mean"] = (
        contrast["auxiliary_score_mean__heldout_truth"]
        - contrast["auxiliary_score_mean__actual_shuffle_top50_union"]
    )
    return components, contrast


def _topk_quality(
    name: str,
    users: np.ndarray,
    topk: np.ndarray,
    prepared: dict,
) -> list[dict]:
    duplicate_rows = 0
    train_pair_count = 0
    ptr = prepared["data"]["csr_ptr"]
    history = prepared["data"]["csr_items"]
    for user, items in zip(users, topk, strict=True):
        duplicate_rows += int(len(set(map(int, items))) != len(items))
        seen = set(map(int, history[ptr[user] : ptr[user + 1]]))
        train_pair_count += sum(int(item) in seen for item in items)
    return [
        {
            "check": f"{name}_top50_duplicate_rows",
            "value": int(duplicate_rows),
            "passed": duplicate_rows == 0,
        },
        {
            "check": f"{name}_top50_train_pairs",
            "value": int(train_pair_count),
            "passed": train_pair_count == 0,
        },
    ]


def run_m3_category_transition_diagnostic(
    cfg: M3CategoryTransitionDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = configure_m3_category_transition_diagnostic(
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
    for arm, model_id in (
        (ARM_ACTUAL, source_runner.ACTUAL_ID),
        (ARM_SHUFFLE, source_runner.SHUFFLE_ID),
    ):
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

    graph_users = sparse_row_similarity(
        graph.user_category_operators[ARM_ACTUAL],
        graph.user_category_operators[ARM_SHUFFLE],
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
        summary = graph_similarity_summary(selected)
        summary.insert(0, "population", population)
        graph_summaries.append(summary)
    graph_summary = pd.concat(graph_summaries, ignore_index=True)
    assignment = clv_assignment_correlation(graph_users)

    views = {
        arm: _model_views(model, prepared, cfg)
        for arm, model in models.items()
    }
    if not np.array_equal(
        views[ARM_ACTUAL]["users"], views[ARM_SHUFFLE]["users"]
    ):
        raise RuntimeError("actual and shuffle evaluation users are misaligned")
    if not np.array_equal(views[ARM_ACTUAL]["users"], eval_users):
        raise RuntimeError("checkpoint recommendations and evaluation cache are misaligned")
    q_eval = graph.clv_percentile[eval_users]
    overlaps = []
    comparisons = (
        (
            "actual_full_vs_shuffle_full",
            views[ARM_ACTUAL]["full_top50"],
            views[ARM_SHUFFLE]["full_top50"],
        ),
        (
            "actual_base_vs_actual_full",
            views[ARM_ACTUAL]["base_top50"],
            views[ARM_ACTUAL]["full_top50"],
        ),
        (
            "shuffle_base_vs_shuffle_full",
            views[ARM_SHUFFLE]["base_top50"],
            views[ARM_SHUFFLE]["full_top50"],
        ),
        (
            "actual_base_vs_shuffle_base",
            views[ARM_ACTUAL]["base_top50"],
            views[ARM_SHUFFLE]["base_top50"],
        ),
    )
    for comparison, reference, model in comparisons:
        overlaps.append(
            recommendation_overlap_summary(
                reference,
                model,
                q_actual=q_eval,
                comparison=comparison,
            )
        )
    recommendation_overlap = pd.concat(overlaps, ignore_index=True)

    populations = _score_pair_population(views, prepared["cache"])
    score_components, score_contrast = _score_tables(
        views, populations, graph.clv_percentile
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
        {
            "check": "actual_graph_active_users",
            "value": int(
                graph.diagnostics["arms"][ARM_ACTUAL]["n_active_users"]
            ),
            "passed": (
                graph.diagnostics["arms"][ARM_ACTUAL]["n_active_users"]
                == prepared["data"]["n_users"]
            ),
        },
    ]
    for arm in (ARM_ACTUAL, ARM_SHUFFLE):
        quality_rows.extend(
            _topk_quality(
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

    overall_graph = graph_summary.loc[
        graph_summary["population"].eq("all_train_users")
        & graph_summary["clv_group"].eq("전체")
    ].iloc[0]
    overall_overlap = recommendation_overlap.loc[
        recommendation_overlap["comparison"].eq(
            "actual_full_vs_shuffle_full"
        )
        & recommendation_overlap["clv_group"].eq("전체")
    ].set_index("k")
    reading = {
        "automatic_model_selection": False,
        "graph_actual_shuffle_mean_jaccard": float(
            overall_graph["mean_edge_jaccard"]
        ),
        "graph_actual_shuffle_median_total_variation_distance": float(
            overall_graph["median_total_variation_distance"]
        ),
        "graph_exact_relation_user_share": float(
            overall_graph["exact_relation_user_share"]
        ),
        "clv_actual_shuffle_overall_spearman": float(
            assignment.loc[
                assignment["scope"].eq("전체"),
                "q_actual_shuffle_spearman",
            ].iloc[0]
        ),
        "actual_shuffle_recommendation_mean_jaccard": {
            f"top{cutoff}": float(overall_overlap.loc[cutoff, "mean_jaccard"])
            for cutoff in (10, 20, 50)
        },
        "interpretation_rule": (
            "inspect exact distances without tuning thresholds: high graph "
            "overlap indicates assignment similarity; low graph overlap with "
            "high recommendation overlap indicates weak graph-to-score transfer"
        ),
        "limitation": (
            "post-hoc on the seen development interval; hypothesis generation only"
        ),
    }

    root = Path(cfg.diagnostic_out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m3_clv_category_transition_failure_diagnostic_{cfg.source_result_id}"
    paths = {
        "graph_user_similarity_csv": root / f"{stem}_graph_users.csv",
        "graph_similarity_summary_csv": root / f"{stem}_graph_summary.csv",
        "clv_assignment_correlation_csv": root / f"{stem}_assignment.csv",
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
            "actual_checkpoint": str(checkpoints[ARM_ACTUAL]),
            "shuffle_checkpoint": str(checkpoints[ARM_SHUFFLE]),
            "input_hash": input_hash,
        },
        "population": {
            "graph": "all 2,500 train users",
            "scores_and_recommendations": (
                "1,230 DAY 684--690 evaluation users; train user-item pairs masked"
            ),
            "heldout_truth_pairs": int(len(populations["heldout_truth"][0])),
        },
        "transformations": [
            "reconstruct train-only actual and degree-matched shuffled CLV relation rows",
            "load immutable seed-42 checkpoints with identity and SHA checks",
            "split each checkpoint score into base dot product and gamma-weighted transition dot product",
            "aggregate graph, score, and recommendation comparisons overall and by train-only CLV quintile",
        ],
        "formulas": {
            "edge_jaccard": "|actual targets ∩ shuffled targets| / |union|",
            "total_variation": "0.5 * sum_d |w_actual(u,d)-w_shuffle(u,d)|",
            "base_score": "z_u_M1 dot z_i_M1",
            "auxiliary_score": "gamma * m_u_transition dot z_i_M1",
            "score_strength": "std(auxiliary score) / std(base score) on the same pairs",
        },
        "validation": quality.to_dict("records"),
        "limitations": [
            "descriptive post-hoc diagnostic on the already seen development interval",
            "no causal, significance, generalization, or model-selection claim",
        ],
    }
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight,
        "source_result": str(source_path),
        "checkpoint_records": records,
        "analysis_ledger": ledger,
        "graph_similarity_summary": graph_summary.to_dict("records"),
        "clv_assignment_correlation": assignment.to_dict("records"),
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
    graph_summary.attrs["score_components"] = score_components
    graph_summary.attrs["score_truth_candidate_contrast"] = score_contrast
    graph_summary.attrs["recommendation_overlap"] = recommendation_overlap
    graph_summary.attrs["clv_assignment_correlation"] = assignment
    graph_summary.attrs["quality_checks"] = quality

    print("\n그래프 actual-shuffle 유사도:")
    print(graph_summary.to_string(index=False))
    print("\n동일 후보집합 점수 분해:")
    print(score_components.to_string(index=False))
    print("\n추천목록 변경:")
    print(recommendation_overlap.to_string(index=False))
    print("\n기술적 판독:")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", graph_summary.attrs["result_paths"])
    return graph_summary


if __name__ == "__main__":
    run_m3_category_transition_diagnostic()
