"""No-retraining diagnostics for the gate-free low-dimensional M2 checkpoint.

The diagnostic answers three descriptive questions with the already-trained
historical-development M1 and M2 checkpoints:

1. Where did held-out truth items move between rank buckets?
2. Which propagated M2 block changes ranking metrics?
3. How large are the effective ID, activity, and transaction-value scores?

No model parameter is updated and no result is used for epoch/model selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
import lightgcn_clv_axis_specific_test10 as test10


CODE_VERSION = "m2-gatefree-lowdim-checkpoint-diagnostic-v1"
VIEW_MODES = ("id_only", "id_n", "id_v", "full")
RANK_BUCKETS = ("1-10", "11-20", "21-50", ">50")
SEGMENT_ORDER = ("저CLV", "중CLV", "고CLV")


@dataclass(frozen=True)
class CheckpointDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m2_checkpoint: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 32


def configure_checkpoint_diagnostic(**overrides) -> CheckpointDiagnosticConfig:
    defaults = gatefree.configure_gatefree_lowdim_run()
    values = {
        "out_dir": defaults.out_dir,
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = CheckpointDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: CheckpointDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "scope": "existing M1 and M2 checkpoints only",
        "split": "historical_development_days_684_690",
        "views": list(VIEW_MODES),
        "rank_buckets": list(RANK_BUCKETS),
        "segments": list(SEGMENT_ORDER),
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
    """Return exact propagated block views without changing the checkpoint."""
    expected = id_dim + 2 * axis_dim
    if user.shape[1] != expected or item.shape[1] != expected:
        raise ValueError(
            f"전파 임베딩 차원이 다릅니다: expected={expected}, "
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


def component_pair_scores(
    user: torch.Tensor,
    item: torch.Tensor,
    users: torch.Tensor,
    items: torch.Tensor,
    *,
    id_dim: int,
    axis_dim: int,
) -> dict[str, torch.Tensor]:
    n_end = id_dim + axis_dim
    selected_user = user.index_select(0, users)
    selected_item = item.index_select(0, items)
    scores = {
        "id": (selected_user[:, :id_dim] * selected_item[:, :id_dim]).sum(1),
        "activity": (
            selected_user[:, id_dim:n_end] * selected_item[:, id_dim:n_end]
        ).sum(1),
        "transaction_value": (
            selected_user[:, n_end:] * selected_item[:, n_end:]
        ).sum(1),
    }
    scores["full"] = scores["id"] + scores["activity"] + scores["transaction_value"]
    return scores


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
    reference_ranks: dict[int, dict[int, int]],
    model_ranks: dict[int, dict[int, int]],
) -> pd.DataFrame:
    counts: dict[tuple[str, str, str], int] = {}
    totals: dict[str, int] = {}
    for user, segment in zip(users.tolist(), segments.tolist(), strict=True):
        for item in np.asarray(truth[int(user)], dtype=np.int64):
            reference_bucket = _rank_bucket(
                int(reference_ranks.get(int(user), {}).get(int(item), 51))
            )
            model_bucket = _rank_bucket(
                int(model_ranks.get(int(user), {}).get(int(item), 51))
            )
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


class _FixedEmbeddingView:
    def __init__(self, user: torch.Tensor, item: torch.Tensor):
        self.user = user
        self.item = item

    def embeddings(self, need_value: bool = True):
        user_zero = self.user.new_zeros((self.user.shape[0], 1))
        item_zero = self.item.new_zeros((self.item.shape[0], 1))
        return self.user, self.item, user_zero, item_zero


def _checkpoint_record(root: Path, model_id: str) -> tuple[Path, dict]:
    candidates = sorted(
        root.rglob(f"{model_id}_s42.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for result_path in candidates:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = Path(payload["checkpoint"])
            if not checkpoint.exists():
                fallback = result_path.with_suffix(".pt")
                checkpoint = fallback if fallback.exists() else checkpoint
            if not checkpoint.exists():
                continue
            expected_sha = payload.get("checkpoint_sha256")
            if expected_sha and file_sha256(checkpoint) != expected_sha:
                continue
            return checkpoint, payload
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(f"{root} 아래에서 검증 가능한 {model_id} checkpoint를 찾지 못했습니다")


def _load_m2(prepared: dict, runner_cfg, diagnostic_cfg):
    if diagnostic_cfg.m2_checkpoint:
        checkpoint = Path(diagnostic_cfg.m2_checkpoint)
        record = {}
    else:
        checkpoint, record = _checkpoint_record(Path(diagnostic_cfg.out_dir), gatefree.MODEL_ID)
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    recorded = payload.get("config", {})
    identity_keys = (
        "dataset",
        "seed",
        "time_cutoff",
        "evaluation_days",
        "id_dim",
        "axis_dim",
        "hidden_dim",
        "axis_budget",
        "n_layers",
        "input_days",
    )
    expected = asdict(runner_cfg)
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
    model, _ = gatefree._build_model(prepared, runner_cfg)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


def _load_m1(prepared: dict, diagnostic_cfg):
    if diagnostic_cfg.m1_checkpoint:
        checkpoint = Path(diagnostic_cfg.m1_checkpoint)
        record = {}
    else:
        checkpoint, record = _checkpoint_record(
            Path(diagnostic_cfg.baseline_result_dir), "m1_64"
        )
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    expected_identity = {
        "model_id": "m1_64",
        "seed": 42,
        "split": "historical_development_days_684_690",
        "final_epoch": 100,
    }
    identity_mismatch = {
        key: {"expected": expected, "actual": record.get(key)}
        for key, expected in expected_identity.items()
        if record and record.get(key) != expected
    }
    recorded_metrics = gatefree._normalise_metric_names(record.get("metrics", {}))
    metric_mismatch = {}
    for metric in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
        expected = prepared["baseline"].get(metric)
        actual = recorded_metrics.get(metric)
        if record and (
            expected is None
            or actual is None
            or not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12)
        ):
            metric_mismatch[metric] = {"expected": expected, "actual": actual}
    if identity_mismatch or metric_mismatch:
        raise RuntimeError(
            "M1 checkpoint가 gate-free 실행에서 재사용한 baseline과 다릅니다: "
            f"identity={identity_mismatch}, metrics={metric_mismatch}"
        )
    x_item, item_cat = v3.item_value_features(
        prepared["data"]["train"], prepared["data"]["n_items"], report=False
    )
    model_cfg = {**prepared["base_cfg"], "DIM": 64}
    model = v3.build_model(
        prepared["data"],
        prepared["data"]["x_val_u"],
        x_item,
        item_cat,
        model_cfg,
    )
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


def _view_metrics(views: dict, prepared: dict) -> pd.DataFrame:
    rows = []
    for view_name in VIEW_MODES:
        user, item = views[view_name]
        metrics, _ = moe._flat_evaluation(
            _FixedEmbeddingView(user, item),
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
def _top50_ranks(
    user: torch.Tensor,
    item: torch.Tensor,
    prepared: dict,
    *,
    batch_size: int,
) -> dict[int, dict[int, int]]:
    cache, data = prepared["cache"], prepared["data"]
    output: dict[int, dict[int, int]] = {}
    for start in range(0, len(cache.users), batch_size):
        batch_users = cache.users[start : start + batch_size]
        indices = torch.as_tensor(batch_users, dtype=torch.long, device=user.device)
        scores = user.index_select(0, indices) @ item.T
        for row, user_id in enumerate(batch_users):
            left, right = data["csr_ptr"][user_id], data["csr_ptr"][user_id + 1]
            if right > left:
                seen = torch.as_tensor(
                    data["csr_items"][left:right],
                    dtype=torch.long,
                    device=user.device,
                )
                scores[row, seen] = -torch.inf
        top50 = scores.topk(50, dim=1).indices.cpu().numpy()
        for user_id, recommended in zip(batch_users.tolist(), top50, strict=True):
            truth_items = set(np.asarray(cache.gt[int(user_id)], dtype=np.int64).tolist())
            output[int(user_id)] = {
                int(item_id): rank
                for rank, item_id in enumerate(recommended.tolist(), start=1)
                if int(item_id) in truth_items
            }
    return output


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
    max_decomposition_error = 0.0
    for start in range(0, len(cache.users), batch_size):
        batch_users = cache.users[start : start + batch_size]
        indices = torch.as_tensor(batch_users, dtype=torch.long, device=user.device)
        component_scores = {
            name: user.index_select(0, indices)[:, block] @ item[:, block].T
            for name, block in slices.items()
        }
        full = user.index_select(0, indices) @ item.T
        max_decomposition_error = max(
            max_decomposition_error,
            float(
                (
                    full
                    - component_scores["id"]
                    - component_scores["activity"]
                    - component_scores["transaction_value"]
                )
                .abs()
                .max()
            ),
        )
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
            0.1 if component in {"activity", "transaction_value"} else np.nan
        )
        row["max_full_decomposition_error"] = max_decomposition_error
        rows.append(row)
    return pd.DataFrame(rows)


def _persist(report: dict, cfg: CheckpointDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "rank_transition_csv": root / "m2_gatefree_rank_transition_by_segment.csv",
        "view_metrics_csv": root / "m2_gatefree_axis_view_metrics.csv",
        "score_strength_csv": root / "m2_gatefree_score_strength.csv",
        "json": root / "m2_gatefree_checkpoint_diagnostic.json",
    }
    test10._atomic_csv(paths["rank_transition_csv"], report["rank_transition"])
    test10._atomic_csv(paths["view_metrics_csv"], report["view_metrics"])
    test10._atomic_csv(paths["score_strength_csv"], report["score_strength"])
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoints only; no training or model selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoints": report["checkpoints"],
        "rank_transition": report["rank_transition"].to_dict("records"),
        "view_metrics": report["view_metrics"].to_dict("records"),
        "score_strength": report["score_strength"].to_dict("records"),
        "interpretation_limits": [
            "descriptive mechanism diagnostic only",
            "no significance claim",
            "does not authorize another look at the protected test split",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_checkpoint_diagnostic(
    cfg: CheckpointDiagnosticConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or configure_checkpoint_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = gatefree.configure_gatefree_lowdim_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = gatefree._prepare(runner_cfg)
    m2_model, m2_checkpoint, _ = _load_m2(prepared, runner_cfg, cfg)
    m1_model, m1_checkpoint, _ = _load_m1(prepared, cfg)
    with torch.no_grad():
        m2_user, m2_item = m2_model.propagate()
        m1_user, m1_item = m1_model.propagate_pref()
    views = axis_views(
        m2_user,
        m2_item,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
    )
    view_metrics = _view_metrics(views, prepared)
    m1_ranks = _top50_ranks(
        m1_user,
        m1_item,
        prepared,
        batch_size=cfg.eval_batch_size,
    )
    m2_ranks = _top50_ranks(
        *views["full"],
        prepared,
        batch_size=cfg.eval_batch_size,
    )
    rank_transition = rank_transition_table(
        users=prepared["cache"].users,
        segments=prepared["cache"].seg,
        truth=prepared["cache"].gt,
        reference_ranks=m1_ranks,
        model_ranks=m2_ranks,
    )
    score_strength = _score_strength(
        m2_user,
        m2_item,
        prepared,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
        batch_size=cfg.eval_batch_size,
    )
    report = {
        "rank_transition": rank_transition,
        "view_metrics": view_metrics,
        "score_strength": score_strength,
        "checkpoints": {
            "m1": {"path": str(m1_checkpoint), "sha256": file_sha256(m1_checkpoint)},
            "m2": {"path": str(m2_checkpoint), "sha256": file_sha256(m2_checkpoint)},
        },
    }
    paths = _persist(report, cfg)
    report["paths"] = paths
    print("\n===== 저·중·고CLV별 정답상품 순위 이동 =====")
    print(rank_transition.to_string(index=False))
    print("\n===== M2 전파 임베딩 블록별 성과 =====")
    print(view_metrics.to_string(index=False))
    print("\n===== 실제 후보점수 영향력 =====")
    print(score_strength.to_string(index=False))
    print("\n결과 파일:", paths)
    return report


if __name__ == "__main__":
    print("No training is started automatically. Call run_checkpoint_diagnostic().")
