"""Checkpoint-only diagnostics for the personal-history candidate-fit M2.

This module never trains or selects a checkpoint.  It reloads the existing
historical-development M1 and M2 checkpoints and describes four mechanisms:

1. exact ID-only / ID+N / ID+V / full ranking metrics,
2. effective ID/N/V candidate-score magnitudes,
3. held-out truth-item rank movements by historical-CLV segment, and
4. popularity and price traits of promoted and displaced Top-10 items.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim_diagnostic as common
import lightgcn_clv_history_item_fit as runner
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-nv-personal-history-candidate-fit-diagnostic-v1"
VIEW_MODES = ("id_only", "id_n", "id_v", "full")
RANK_BUCKETS = ("1-10", "11-20", "21-50", ">50")
SEGMENT_ORDER = ("저CLV", "중CLV", "고CLV")
ITEM_ROLES = (
    "test_truth_new_items",
    "m1_top10",
    "m2_top10",
    "m2_promoted_top10",
    "m1_displaced_top10",
)


@dataclass(frozen=True)
class HistoryItemFitDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m2_checkpoint: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 32
    top_product_examples: int = 20


def configure_history_item_fit_diagnostic(
    **overrides,
) -> HistoryItemFitDiagnosticConfig:
    defaults = runner.configure_history_item_fit_run()
    values = {
        "out_dir": defaults.out_dir,
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = HistoryItemFitDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    if cfg.eval_batch_size <= 0 or cfg.top_product_examples <= 0:
        raise ValueError("eval_batch_size와 top_product_examples는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: HistoryItemFitDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "checkpoint_selection": False,
        "scope": "existing seed-42 historical-development M1 and M2 checkpoints",
        "split": "historical_development_days_684_690",
        "views": list(VIEW_MODES),
        "rank_buckets": list(RANK_BUCKETS),
        "segments": list(SEGMENT_ORDER),
        "item_roles": list(ITEM_ROLES),
        "candidate_score_scope": "all unseen evaluation candidates",
        "statistical_note": "descriptive checkpoint diagnostic; no significance claim",
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def axis_views(
    user: torch.Tensor,
    item: torch.Tensor,
    *,
    id_dim: int,
    axis_dim: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Slice the exact trained ID/N/V score blocks without retraining."""
    expected = id_dim + 2 * axis_dim
    if user.shape[1] != expected or item.shape[1] != expected:
        raise ValueError(
            f"임베딩 차원이 다릅니다: expected={expected}, "
            f"user={user.shape[1]}, item={item.shape[1]}"
        )
    n_end = id_dim + axis_dim
    return {
        "id_only": (user[:, :id_dim], item[:, :id_dim]),
        "id_n": (user[:, :n_end], item[:, :n_end]),
        "id_v": (
            torch.cat([user[:, :id_dim], user[:, n_end:]], dim=1),
            torch.cat([item[:, :id_dim], item[:, n_end:]], dim=1),
        ),
        "full": (user, item),
    }


def _rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def rank_transition_table(
    *,
    users: np.ndarray,
    segments: np.ndarray,
    truth: dict[int, np.ndarray],
    reference_top50: np.ndarray,
    model_top50: np.ndarray,
) -> pd.DataFrame:
    counts: dict[tuple[str, str, str], int] = {}
    totals: dict[str, int] = {}
    for row, (user, segment) in enumerate(
        zip(users.tolist(), segments.tolist(), strict=True)
    ):
        reference_rank = {
            int(item): rank
            for rank, item in enumerate(reference_top50[row].tolist(), start=1)
        }
        model_rank = {
            int(item): rank
            for rank, item in enumerate(model_top50[row].tolist(), start=1)
        }
        for item in np.asarray(truth[int(user)], dtype=np.int64):
            reference_bucket = _rank_bucket(reference_rank.get(int(item), 51))
            model_bucket = _rank_bucket(model_rank.get(int(item), 51))
            key = (str(segment), reference_bucket, model_bucket)
            counts[key] = counts.get(key, 0) + 1
            totals[str(segment)] = totals.get(str(segment), 0) + 1
    rows = []
    for segment in SEGMENT_ORDER:
        for reference_bucket in RANK_BUCKETS:
            for model_bucket in RANK_BUCKETS:
                count = counts.get((segment, reference_bucket, model_bucket), 0)
                if count == 0:
                    continue
                rows.append(
                    {
                        "segment": segment,
                        "reference_bucket": reference_bucket,
                        "model_bucket": model_bucket,
                        "truth_item_count": count,
                        "share_within_segment": count / totals[segment],
                    }
                )
    return pd.DataFrame(rows)


def _load_m2(prepared: dict, runner_cfg, diagnostic_cfg):
    if diagnostic_cfg.m2_checkpoint:
        checkpoint = Path(diagnostic_cfg.m2_checkpoint)
        record = {}
    else:
        checkpoint, record = common._checkpoint_record(
            Path(diagnostic_cfg.out_dir), runner.MODEL_ID
        )
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    recorded = payload.get("config", {})
    expected = asdict(runner_cfg)
    identity_keys = (
        "dataset",
        "seed",
        "time_cutoff",
        "evaluation_days",
        "epochs",
        "id_dim",
        "axis_dim",
        "rho",
        "n_layers",
        "batch_size",
        "lr",
        "pref_reg",
        "input_days",
    )
    mismatch = {
        key: {"expected": expected[key], "actual": recorded.get(key)}
        for key in identity_keys
        if recorded.get(key) != expected[key]
    }
    if payload.get("input_hash") != prepared["input_hash"]:
        mismatch["input_hash"] = {
            "expected": prepared["input_hash"],
            "actual": payload.get("input_hash"),
        }
    if mismatch:
        raise RuntimeError(f"M2 checkpoint identity mismatch: {mismatch}")
    model, _ = runner._build_model(prepared, runner_cfg)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


def _view_metrics(
    m1_user: torch.Tensor,
    m1_item: torch.Tensor,
    views: dict[str, tuple[torch.Tensor, torch.Tensor]],
    prepared: dict,
) -> pd.DataFrame:
    rows = []
    named_views = {"m1_64": (m1_user, m1_item), **views}
    for view_name, (user, item) in named_views.items():
        metrics, _ = moe._flat_evaluation(
            common._FixedEmbeddingView(user, item),
            0.0,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=False,
        )
        rows.append({"view": view_name, **test10._public_metrics(metrics)})
    return pd.DataFrame(rows)


@torch.no_grad()
def _masked_topk(
    user: torch.Tensor,
    item: torch.Tensor,
    prepared: dict,
    *,
    max_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    users = prepared["cache"].users.astype(np.int64)
    topk = np.empty((len(users), max_k), np.int32)
    csr_ptr = prepared["data"]["csr_ptr"]
    csr_items = prepared["data"]["csr_items"]
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        tensor_users = torch.as_tensor(
            batch_users, dtype=torch.long, device=user.device
        )
        scores = user.index_select(0, tensor_users) @ item.T
        for row, user_id in enumerate(batch_users):
            left, right = csr_ptr[user_id], csr_ptr[user_id + 1]
            if right > left:
                scores[row, csr_items[left:right]] = -torch.inf
        topk[start : start + len(batch_users)] = (
            scores.topk(max_k, dim=1).indices.cpu().numpy()
        )
    return users, topk


class _RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.total_abs = 0.0

    def update(self, values: torch.Tensor):
        values = values.double()
        self.count += values.numel()
        self.total += float(values.sum())
        self.total_square += float(values.square().sum())
        self.total_abs += float(values.abs().sum())

    def finish(self) -> dict:
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "candidate_pair_count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "mean_abs": self.total_abs / self.count,
        }


@torch.no_grad()
def _score_strength(
    user: torch.Tensor,
    item: torch.Tensor,
    prepared: dict,
    *,
    id_dim: int,
    axis_dim: int,
    rho: float,
    batch_size: int,
) -> pd.DataFrame:
    cache, data = prepared["cache"], prepared["data"]
    n_end = id_dim + axis_dim
    slices = {
        "id": slice(0, id_dim),
        "activity": slice(id_dim, n_end),
        "transaction_value": slice(n_end, n_end + axis_dim),
    }
    moments = {name: _RunningMoments() for name in slices}
    full_moments = _RunningMoments()
    max_error = 0.0
    for start in range(0, len(cache.users), batch_size):
        batch_users = cache.users[start : start + batch_size]
        indices = torch.as_tensor(batch_users, dtype=torch.long, device=user.device)
        selected_user = user.index_select(0, indices)
        component_scores = {
            name: selected_user[:, block] @ item[:, block].T
            for name, block in slices.items()
        }
        full = selected_user @ item.T
        reconstructed = sum(component_scores.values())
        max_error = max(max_error, float((full - reconstructed).abs().max()))
        for row, user_id in enumerate(batch_users):
            unseen = torch.ones(item.shape[0], dtype=torch.bool, device=user.device)
            left, right = data["csr_ptr"][user_id], data["csr_ptr"][user_id + 1]
            if right > left:
                seen = torch.as_tensor(
                    data["csr_items"][left:right],
                    dtype=torch.long,
                    device=user.device,
                )
                unseen[seen] = False
            for name in slices:
                moments[name].update(component_scores[name][row, unseen])
            full_moments.update(full[row, unseen])
    statistics = {name: value.finish() for name, value in moments.items()}
    statistics["full"] = full_moments.finish()
    id_std = statistics["id"]["std"]
    rows = []
    for component in ("id", "activity", "transaction_value", "full"):
        row = {"component": component, **statistics[component]}
        row["std_ratio_to_id"] = row["std"] / id_std if id_std > 0 else np.nan
        row["nominal_axis_coefficient"] = (
            rho if component in {"activity", "transaction_value"} else np.nan
        )
        row["coefficient_note"] = (
            "already included through sqrt(rho) on both embedding sides"
            if component in {"activity", "transaction_value"}
            else ""
        )
        row["max_full_decomposition_error"] = max_error
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_item_traits(train: pd.DataFrame, n_items: int) -> pd.DataFrame:
    def modal(series):
        mode = series.mode(dropna=True)
        return mode.iat[0] if len(mode) else "UNKNOWN"

    basket_column = "b_raw" if "b_raw" in train else "t"
    item = train.groupby("i_idx", sort=True).agg(
        item_id=("i_raw", "first"),
        category=("cat_raw", modal),
        train_row_count=("i_idx", "size"),
        train_user_count=("u_idx", "nunique"),
        train_basket_count=(basket_column, "nunique"),
        mean_unit_price=("up", "mean"),
    )
    item["price_percentile"] = item["mean_unit_price"].rank(
        pct=True, method="average"
    )
    result = pd.DataFrame({"item_idx": np.arange(n_items, dtype=int)})
    return result.merge(item.reset_index(), on="item_idx", how="left")


def item_role_occurrences(
    *,
    users: np.ndarray,
    segments: np.ndarray,
    truth: dict[int, np.ndarray],
    m1_top50: np.ndarray,
    m2_top50: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for row, (user, segment) in enumerate(
        zip(users.tolist(), segments.tolist(), strict=True)
    ):
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        truth_set = set(truth_items.tolist())
        m1_top10 = m1_top50[row, :10]
        m2_top10 = m2_top50[row, :10]
        m1_set = set(m1_top10.tolist())
        m2_set = set(m2_top10.tolist())
        roles = {
            "test_truth_new_items": truth_items,
            "m1_top10": m1_top10,
            "m2_top10": m2_top10,
            "m2_promoted_top10": np.asarray(
                [item for item in m2_top10 if int(item) not in m1_set],
                dtype=np.int64,
            ),
            "m1_displaced_top10": np.asarray(
                [item for item in m1_top10 if int(item) not in m2_set],
                dtype=np.int64,
            ),
        }
        for role, items in roles.items():
            rank_source = m2_top10 if role.startswith("m2_") else m1_top10
            rank = {int(item): position for position, item in enumerate(rank_source, 1)}
            for item in items:
                rows.append(
                    {
                        "user_idx": int(user),
                        "segment": str(segment),
                        "role": role,
                        "item_idx": int(item),
                        "rank_if_top10": rank.get(int(item), np.nan),
                        "is_truth_item": int(item) in truth_set,
                    }
                )
    return pd.DataFrame(rows)


def summarize_item_roles(
    occurrences: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = occurrences.merge(item_traits, on="item_idx", how="left")
    summary = (
        merged.groupby(["segment", "role"], sort=False, dropna=False)
        .agg(
            item_occurrence_count=("item_idx", "size"),
            distinct_item_count=("item_idx", "nunique"),
            truth_hit_share=("is_truth_item", "mean"),
            mean_train_user_count=("train_user_count", "mean"),
            median_train_user_count=("train_user_count", "median"),
            mean_train_basket_count=("train_basket_count", "mean"),
            mean_unit_price=("mean_unit_price", "mean"),
            mean_price_percentile=("price_percentile", "mean"),
        )
        .reset_index()
    )
    return merged, summary


def _product_examples(
    merged: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    focused = merged[merged.role.isin(["m2_promoted_top10", "m1_displaced_top10"])]
    if focused.empty:
        return focused.copy()
    examples = (
        focused.groupby(
            ["segment", "role", "item_idx"], sort=False, dropna=False
        )
        .agg(
            occurrence_count=("item_idx", "size"),
            affected_user_count=("user_idx", "nunique"),
            truth_hit_count=("is_truth_item", "sum"),
            item_id=("item_id", "first"),
            category=("category", "first"),
            train_user_count=("train_user_count", "first"),
            train_basket_count=("train_basket_count", "first"),
            mean_unit_price=("mean_unit_price", "first"),
            price_percentile=("price_percentile", "first"),
        )
        .reset_index()
    )
    return (
        examples.sort_values(
            ["segment", "role", "occurrence_count", "truth_hit_count"],
            ascending=[True, True, False, False],
        )
        .groupby(["segment", "role"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _persist(report: dict, cfg: HistoryItemFitDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "view_metrics_csv": root / "m2_history_item_fit_axis_view_metrics.csv",
        "score_strength_csv": root / "m2_history_item_fit_score_strength.csv",
        "rank_transition_csv": root / "m2_history_item_fit_rank_transition_by_segment.csv",
        "item_role_occurrences_csv": root / "m2_history_item_fit_item_role_occurrences.csv",
        "item_role_summary_csv": root / "m2_history_item_fit_item_role_summary.csv",
        "product_examples_csv": root / "m2_history_item_fit_product_examples.csv",
        "json": root / "m2_history_item_fit_checkpoint_diagnostic.json",
    }
    for key, frame_key in (
        ("view_metrics_csv", "view_metrics"),
        ("score_strength_csv", "score_strength"),
        ("rank_transition_csv", "rank_transition"),
        ("item_role_occurrences_csv", "item_role_occurrences"),
        ("item_role_summary_csv", "item_role_summary"),
        ("product_examples_csv", "product_examples"),
    ):
        test10._atomic_csv(paths[key], report[frame_key])
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoints only; no training or model selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoints": report["checkpoints"],
        "view_metrics": report["view_metrics"].to_dict("records"),
        "score_strength": report["score_strength"].to_dict("records"),
        "rank_transition": report["rank_transition"].to_dict("records"),
        "item_role_summary": report["item_role_summary"].to_dict("records"),
        "product_examples": report["product_examples"].to_dict("records"),
        "interpretation_limits": [
            "descriptive mechanism diagnostic only",
            "no significance claim",
            "does not train, select, or tune a model",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_history_item_fit_diagnostic(
    cfg: HistoryItemFitDiagnosticConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or configure_history_item_fit_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = runner.configure_history_item_fit_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = runner._prepare(runner_cfg)
    m2_model, m2_checkpoint, _ = _load_m2(prepared, runner_cfg, cfg)
    m1_model, m1_checkpoint, _ = common._load_m1(prepared, cfg)
    with torch.no_grad():
        m2_user, m2_item, *_ = m2_model.embeddings(need_value=False)
        m1_user, m1_item, *_ = m1_model.embeddings(need_value=False)
    views = axis_views(
        m2_user,
        m2_item,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
    )
    view_metrics = _view_metrics(m1_user, m1_item, views, prepared)
    users, m1_top50 = _masked_topk(
        m1_user,
        m1_item,
        prepared,
        max_k=50,
        batch_size=cfg.eval_batch_size,
    )
    m2_users, m2_top50 = _masked_topk(
        *views["full"],
        prepared,
        max_k=50,
        batch_size=cfg.eval_batch_size,
    )
    if not np.array_equal(users, m2_users):
        raise RuntimeError("M1과 M2 평가 사용자 순서가 다릅니다")
    rank_transition = rank_transition_table(
        users=users,
        segments=prepared["cache"].seg,
        truth=prepared["cache"].gt,
        reference_top50=m1_top50,
        model_top50=m2_top50,
    )
    score_strength = _score_strength(
        m2_user,
        m2_item,
        prepared,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
        rho=runner_cfg.rho,
        batch_size=cfg.eval_batch_size,
    )
    occurrences = item_role_occurrences(
        users=users,
        segments=prepared["cache"].seg,
        truth=prepared["cache"].gt,
        m1_top50=m1_top50,
        m2_top50=m2_top50,
    )
    item_traits = _raw_item_traits(
        prepared["data"]["train"], prepared["data"]["n_items"]
    )
    merged_occurrences, item_role_summary = summarize_item_roles(
        occurrences, item_traits
    )
    product_examples = _product_examples(
        merged_occurrences,
        top_n=cfg.top_product_examples,
    )
    report = {
        "view_metrics": view_metrics,
        "score_strength": score_strength,
        "rank_transition": rank_transition,
        "item_role_occurrences": merged_occurrences,
        "item_role_summary": item_role_summary,
        "product_examples": product_examples,
        "checkpoints": {
            "m1": {"path": str(m1_checkpoint), "sha256": file_sha256(m1_checkpoint)},
            "m2": {"path": str(m2_checkpoint), "sha256": file_sha256(m2_checkpoint)},
        },
    }
    report["paths"] = _persist(report, cfg)

    print("\n===== 1) ID/N/V 블록별 성과 =====")
    print(view_metrics.to_string(index=False))
    print("\n===== 2) 실제 후보점수 영향력 =====")
    print(score_strength.to_string(index=False))
    print("\n===== 3) 저·중·고CLV별 정답상품 순위 이동 =====")
    print(rank_transition.to_string(index=False))
    print("\n===== 4) 추천 역할별 상품 인기도·가격 특성 =====")
    print(item_role_summary.to_string(index=False))
    print("\n===== 5) 새로 올라오거나 밀려난 상품 예시 =====")
    print(product_examples.to_string(index=False))
    print("\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print("No training is started automatically. Call run_history_item_fit_diagnostic().")
