"""Checkpoint-only diagnostics for the dynamic historical-CLV M2.

No model is trained or selected here.  The diagnostic reloads the matched
multi-anchor rho=0 checkpoint and the dynamic-CLV checkpoint, then separates:

1. the total M1-to-M2 ranking change,
2. the jointly-trained parameter-path change inside M2 when rho is set to 0,
3. the direct forward effect of the CLV condition at fixed M2 parameters, and
4. the amount of historical-CLV change that was actually present across anchors.
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
import lightgcn_clv_dynamic_multianchor as runner
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-dynamic-clv-multianchor-checkpoint-diagnostic-v1"
MODEL_IDS = ("m1_multianchor_rho0", "m2_dynamic_clv")
COMPARISONS = {
    "m1_vs_m2_total": ("m1", "m2"),
    "m1_vs_m2_rho0_joint_training": ("m1", "m2_rho0"),
    "m2_rho0_vs_m2_direct_clv": ("m2_rho0", "m2"),
}


@dataclass(frozen=True)
class DynamicMultiAnchorDiagnosticConfig:
    out_dir: str = ""
    m1_checkpoint: str = ""
    m2_checkpoint: str = ""
    eval_batch_size: int = 32
    ks: tuple[int, ...] = (10, 20, 50)


def configure_dynamic_multianchor_diagnostic(
    **overrides,
) -> DynamicMultiAnchorDiagnosticConfig:
    source = runner.configure_dynamic_multianchor()
    values = {"out_dir": source.out_dir}
    values.update(overrides)
    cfg = DynamicMultiAnchorDiagnosticConfig(**values)
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size는 양수여야 합니다")
    if not cfg.ks or any(k <= 0 for k in cfg.ks):
        raise ValueError("ks는 양의 정수여야 합니다")
    if tuple(sorted(set(cfg.ks))) != cfg.ks:
        raise ValueError("ks는 중복 없는 오름차순이어야 합니다")
    return cfg


def preflight_summary(cfg: DynamicMultiAnchorDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "checkpoint_selection": False,
        "split": "historical_development_days_684_690",
        "models": list(MODEL_IDS),
        "views": ["m1", "m2_rho0", "m2"],
        "comparisons": list(COMPARISONS),
        "ks": list(cfg.ks),
        "candidate_score_scope": "all unseen evaluation candidates",
        "interpretation": {
            "m1_vs_m2_total": "total learned and direct M2 effect",
            "m1_vs_m2_rho0_joint_training": (
                "M2 parameter trajectory with the direct CLV condition disabled"
            ),
            "m2_rho0_vs_m2_direct_clv": (
                "direct CLV forward effect at fixed trained M2 parameters"
            ),
        },
        "statistical_note": (
            "descriptive seed-42 checkpoint diagnostic; no significance claim"
        ),
        "out_dir": cfg.out_dir,
    }


class _RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.total_abs = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double()
        self.count += int(values.numel())
        self.total += float(values.sum())
        self.total_square += float(values.square().sum())
        self.total_abs += float(values.abs().sum())

    def finish(self) -> dict[str, float | int]:
        if self.count == 0:
            raise RuntimeError("candidate score diagnostic received no values")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "candidate_pair_count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "mean_abs": self.total_abs / self.count,
        }


def _score_frame_from_moments(
    moments: dict[str, _RunningMoments],
    *,
    max_decomposition_error: float,
) -> pd.DataFrame:
    stats = {name: value.finish() for name, value in moments.items()}
    reference_std = float(stats["reference_m1"]["std"])
    rows = []
    for component in (
        "reference_m1",
        "joint_training_path",
        "direct_clv_condition",
        "total_m2_minus_m1",
    ):
        row = {"component": component, **stats[component]}
        row["std_ratio_to_m1"] = (
            float(row["std"]) / reference_std if reference_std > 0 else np.nan
        )
        row["max_decomposition_error"] = max_decomposition_error
        rows.append(row)
    return pd.DataFrame(rows)


def score_component_statistics(
    m1_scores: torch.Tensor,
    m2_rho0_scores: torch.Tensor,
    m2_scores: torch.Tensor,
) -> pd.DataFrame:
    """Summarize the exact total = training-path + direct-CLV decomposition."""
    if not (
        m1_scores.shape == m2_rho0_scores.shape == m2_scores.shape
    ):
        raise ValueError("score tensors must have the same shape")
    joint = m2_rho0_scores - m1_scores
    direct = m2_scores - m2_rho0_scores
    total = m2_scores - m1_scores
    moments = {
        "reference_m1": _RunningMoments(),
        "joint_training_path": _RunningMoments(),
        "direct_clv_condition": _RunningMoments(),
        "total_m2_minus_m1": _RunningMoments(),
    }
    for name, values in (
        ("reference_m1", m1_scores),
        ("joint_training_path", joint),
        ("direct_clv_condition", direct),
        ("total_m2_minus_m1", total),
    ):
        moments[name].update(values)
    error = float((total - joint - direct).abs().max())
    return _score_frame_from_moments(
        moments, max_decomposition_error=error
    )


def topk_change_table(
    *,
    users: np.ndarray,
    segments: np.ndarray,
    topk_by_view: dict[str, np.ndarray],
    ks: tuple[int, ...],
) -> pd.DataFrame:
    if set(topk_by_view) != {"m1", "m2_rho0", "m2"}:
        raise ValueError("topk_by_view must contain m1, m2_rho0, and m2")
    if len(users) != len(segments):
        raise ValueError("users and segments must have the same length")
    if any(len(values) != len(users) for values in topk_by_view.values()):
        raise ValueError("all top-k arrays must match the user count")

    segment_values = ["전체", *pd.unique(segments).tolist()]
    rows = []
    for segment in segment_values:
        selected = (
            np.ones(len(users), dtype=bool)
            if segment == "전체"
            else segments.astype(str) == str(segment)
        )
        for k in ks:
            for comparison, (left_name, right_name) in COMPARISONS.items():
                left = topk_by_view[left_name][selected, :k]
                right = topk_by_view[right_name][selected, :k]
                intersections = np.empty(len(left), dtype=np.int32)
                for row_index in range(len(left)):
                    intersections[row_index] = len(
                        set(left[row_index].tolist())
                        & set(right[row_index].tolist())
                    )
                changed_count = k - intersections
                union = 2 * k - intersections
                rows.append(
                    {
                        "segment": str(segment),
                        "k": int(k),
                        "comparison": comparison,
                        "user_count": int(selected.sum()),
                        "changed_user_count": int((changed_count > 0).sum()),
                        "changed_user_share": float((changed_count > 0).mean()),
                        "mean_changed_item_count": float(changed_count.mean()),
                        "mean_jaccard": float((intersections / union).mean()),
                    }
                )
    return pd.DataFrame(rows)


def condition_variation_table(
    conditions: np.ndarray,
    *,
    valid: np.ndarray,
    scope: str,
) -> pd.DataFrame:
    conditions = np.asarray(conditions, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if conditions.ndim != 2 or conditions.shape[1] != len(valid):
        raise ValueError("conditions must be context-by-user and match valid")
    if conditions.shape[0] < 2 or not valid.any():
        raise ValueError("at least two contexts and one valid user are required")
    values = conditions[:, valid]
    user_std = values.std(axis=0)
    user_range = values.max(axis=0) - values.min(axis=0)
    first_last = np.abs(values[-1] - values[0])
    correlations = []
    for index in range(values.shape[0] - 1):
        left, right = values[index], values[index + 1]
        if left.std() > 0 and right.std() > 0:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    row = {
        "scope": scope,
        "context_count": int(values.shape[0]),
        "valid_user_count": int(values.shape[1]),
        "condition_std_across_users_mean": float(values.std(axis=1).mean()),
        "mean_user_time_std": float(user_std.mean()),
        "median_user_time_std": float(np.median(user_std)),
        "mean_user_range": float(user_range.mean()),
        "median_user_range": float(np.median(user_range)),
        "mean_abs_first_last_change": float(first_last.mean()),
        "median_abs_first_last_change": float(np.median(first_last)),
        "unchanged_user_share": float((user_range <= 1e-8).mean()),
        "changed_gt_0_01_user_share": float((user_range > 0.01).mean()),
        "changed_gt_0_05_user_share": float((user_range > 0.05).mean()),
        "changed_gt_0_10_user_share": float((user_range > 0.10).mean()),
        "mean_adjacent_user_correlation": (
            float(np.mean(correlations)) if correlations else np.nan
        ),
    }
    return pd.DataFrame([row])


def rank_boundary_table(
    *,
    ks: tuple[int, ...],
    margins: dict[int, np.ndarray],
    direct_shift_ranges: np.ndarray,
    direct_changed: dict[int, np.ndarray],
    total_changed: dict[int, np.ndarray],
) -> pd.DataFrame:
    direct_shift_ranges = np.asarray(direct_shift_ranges, dtype=np.float64)
    rows = []
    for k in ks:
        margin = np.asarray(margins[k], dtype=np.float64)
        if len(margin) != len(direct_shift_ranges):
            raise ValueError("margin and score-shift arrays must align")
        ratio = np.divide(
            direct_shift_ranges,
            margin,
            out=np.full_like(direct_shift_ranges, np.inf),
            where=margin > 1e-12,
        )
        ratio[(margin <= 1e-12) & (direct_shift_ranges <= 1e-12)] = 0.0
        rows.append(
            {
                "k": int(k),
                "user_count": int(len(margin)),
                "mean_reference_k_kplus1_margin": float(margin.mean()),
                "median_reference_k_kplus1_margin": float(np.median(margin)),
                "mean_direct_score_shift_range": float(
                    direct_shift_ranges.mean()
                ),
                "median_direct_score_shift_range": float(
                    np.median(direct_shift_ranges)
                ),
                "median_shift_range_to_margin": float(np.median(ratio)),
                "shift_range_ge_margin_user_share": float(
                    (direct_shift_ranges >= margin).mean()
                ),
                "direct_topk_changed_user_share": float(
                    np.asarray(direct_changed[k], dtype=bool).mean()
                ),
                "total_topk_changed_user_share": float(
                    np.asarray(total_changed[k], dtype=bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _find_arm_record(
    cfg: DynamicMultiAnchorDiagnosticConfig, model_id: str
) -> tuple[Path, dict]:
    explicit = cfg.m1_checkpoint if model_id == MODEL_IDS[0] else cfg.m2_checkpoint
    if explicit:
        checkpoint = Path(explicit)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint, {}
    records = sorted(Path(cfg.out_dir).glob(f"arms/*/{model_id}_s42.json"))
    if len(records) != 1:
        raise RuntimeError(
            f"{model_id} completed arm record must be unique; found={records}"
        )
    record = json.loads(records[0].read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise RuntimeError(f"{model_id} checkpoint hash mismatch")
    return checkpoint, record


def _load_model(
    prepared: dict,
    source_cfg: runner.DynamicMultiAnchorConfig,
    diagnostic_cfg: DynamicMultiAnchorDiagnosticConfig,
    model_id: str,
    *,
    rho: float,
):
    checkpoint, record = _find_arm_record(diagnostic_cfg, model_id)
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    expected = asdict(source_cfg)
    recorded = payload.get("config", {})
    identity_keys = (
        "dataset", "seed", "time_cutoff", "evaluation_days", "anchor_count",
        "anchor_horizon_days", "epochs", "embedding_dim", "n_layers", "rho",
        "batch_size", "lr", "pref_reg",
    )
    mismatch = {
        key: {"expected": expected[key], "actual": recorded.get(key)}
        for key in identity_keys
        if recorded.get(key) != expected[key]
    }
    if payload.get("model_id") != model_id:
        mismatch["model_id"] = {
            "expected": model_id,
            "actual": payload.get("model_id"),
        }
    if mismatch:
        raise RuntimeError(f"{model_id} checkpoint identity mismatch: {mismatch}")
    model = runner._build_model(prepared, source_cfg, rho)
    model.load_state_dict(payload["state"], strict=True)
    model.set_context(prepared["final_context"]["name"])
    model.eval()
    return model, checkpoint, record


def _changed_rows(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    return np.any(
        np.sort(left[:, :k], axis=1) != np.sort(right[:, :k], axis=1),
        axis=1,
    )


@torch.no_grad()
def _candidate_diagnostics(
    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    prepared: dict,
    cfg: DynamicMultiAnchorDiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = prepared["cache"].users.astype(np.int64)
    max_k = max(cfg.ks)
    if prepared["data"]["n_items"] <= max_k:
        raise RuntimeError("not enough items for requested top-k diagnostic")
    top_indices = {
        name: np.empty((len(users), max_k), dtype=np.int32)
        for name in embeddings
    }
    margins = {k: np.empty(len(users), dtype=np.float64) for k in cfg.ks}
    direct_shift_ranges = np.empty(len(users), dtype=np.float64)
    moments = {
        "reference_m1": _RunningMoments(),
        "joint_training_path": _RunningMoments(),
        "direct_clv_condition": _RunningMoments(),
        "total_m2_minus_m1": _RunningMoments(),
    }
    max_error = 0.0
    csr_ptr = prepared["data"]["csr_ptr"]
    csr_items = prepared["data"]["csr_items"]

    for start in range(0, len(users), cfg.eval_batch_size):
        stop = min(start + cfg.eval_batch_size, len(users))
        batch_users = users[start:stop]
        device = embeddings["m1"][0].device
        selected = torch.as_tensor(batch_users, dtype=torch.long, device=device)
        scores = {
            name: user.index_select(0, selected) @ item.T
            for name, (user, item) in embeddings.items()
        }
        unseen = torch.ones_like(scores["m1"], dtype=torch.bool)
        for row, user_id in enumerate(batch_users):
            left, right = csr_ptr[user_id], csr_ptr[user_id + 1]
            if right > left:
                seen = torch.as_tensor(
                    csr_items[left:right], dtype=torch.long, device=device
                )
                unseen[row, seen] = False

        joint = scores["m2_rho0"] - scores["m1"]
        direct = scores["m2"] - scores["m2_rho0"]
        total = scores["m2"] - scores["m1"]
        max_error = max(
            max_error, float((total - joint - direct).abs().max())
        )
        for name, values in (
            ("reference_m1", scores["m1"]),
            ("joint_training_path", joint),
            ("direct_clv_condition", direct),
            ("total_m2_minus_m1", total),
        ):
            moments[name].update(values[unseen])

        direct_max = direct.masked_fill(~unseen, -torch.inf).max(dim=1).values
        direct_min = direct.masked_fill(~unseen, torch.inf).min(dim=1).values
        direct_shift_ranges[start:stop] = (
            direct_max - direct_min
        ).cpu().numpy()
        for name, values in scores.items():
            masked = values.masked_fill(~unseen, -torch.inf)
            top_values, indices = masked.topk(max_k + 1, dim=1)
            top_indices[name][start:stop] = indices[:, :max_k].cpu().numpy()
            if name == "m2_rho0":
                top_values = top_values.cpu().numpy()
                for k in cfg.ks:
                    margins[k][start:stop] = top_values[:, k - 1] - top_values[:, k]

    score_effect = _score_frame_from_moments(
        moments, max_decomposition_error=max_error
    )
    topk_change = topk_change_table(
        users=users,
        segments=prepared["cache"].seg,
        topk_by_view=top_indices,
        ks=cfg.ks,
    )
    direct_changed = {
        k: _changed_rows(top_indices["m2_rho0"], top_indices["m2"], k)
        for k in cfg.ks
    }
    total_changed = {
        k: _changed_rows(top_indices["m1"], top_indices["m2"], k)
        for k in cfg.ks
    }
    boundary = rank_boundary_table(
        ks=cfg.ks,
        margins=margins,
        direct_shift_ranges=direct_shift_ranges,
        direct_changed=direct_changed,
        total_changed=total_changed,
    )
    return topk_change, score_effect, boundary


def _condition_diagnostics(prepared: dict) -> pd.DataFrame:
    training_conditions = np.stack(
        [context["condition"] for context in prepared["contexts"]]
    )
    training_valid = np.logical_and.reduce(
        [np.isfinite(context["clv_proxy"]) for context in prepared["contexts"]]
    )
    all_conditions = np.concatenate(
        [
            training_conditions,
            prepared["final_context"]["condition"][None, :],
        ],
        axis=0,
    )
    all_valid = training_valid & np.isfinite(prepared["final_clv"]["clv_proxy"])
    return pd.concat(
        [
            condition_variation_table(
                training_conditions,
                valid=training_valid,
                scope="four_training_anchors",
            ),
            condition_variation_table(
                all_conditions,
                valid=all_valid,
                scope="training_anchors_plus_evaluation",
            ),
        ],
        ignore_index=True,
    )


def _persist(report: dict, cfg: DynamicMultiAnchorDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics" / "dynamic_clv_multianchor"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "topk_change_csv": root / "dynamic_clv_topk_change.csv",
        "score_effect_csv": root / "dynamic_clv_score_effect.csv",
        "rank_boundary_csv": root / "dynamic_clv_rank_boundary.csv",
        "condition_variation_csv": root / "dynamic_clv_condition_variation.csv",
        "json": root / "dynamic_clv_checkpoint_diagnostic.json",
    }
    for key, frame_key in (
        ("topk_change_csv", "topk_change"),
        ("score_effect_csv", "score_effect"),
        ("rank_boundary_csv", "rank_boundary"),
        ("condition_variation_csv", "condition_variation"),
    ):
        test10._atomic_csv(paths[key], report[frame_key])
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoints only; no training or model selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoints": report["checkpoints"],
        "topk_change": report["topk_change"].to_dict("records"),
        "score_effect": report["score_effect"].to_dict("records"),
        "rank_boundary": report["rank_boundary"].to_dict("records"),
        "condition_variation": report["condition_variation"].to_dict("records"),
        "interpretation_limits": [
            "descriptive mechanism diagnostic only",
            "m2_rho0 is an evaluation-time ablation at fixed M2 parameters",
            "no checkpoint selection, tuning, or significance claim",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_dynamic_multianchor_diagnostic(
    cfg: DynamicMultiAnchorDiagnosticConfig | None = None,
) -> dict:
    cfg = cfg or configure_dynamic_multianchor_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    source_cfg = runner.configure_dynamic_multianchor(out_dir=cfg.out_dir)
    prepared = runner._prepare(source_cfg)
    m1, m1_checkpoint, _ = _load_model(
        prepared, source_cfg, cfg, MODEL_IDS[0], rho=0.0
    )
    m2, m2_checkpoint, _ = _load_model(
        prepared, source_cfg, cfg, MODEL_IDS[1], rho=source_cfg.rho
    )
    m2_rho0, _, _ = _load_model(
        prepared, source_cfg, cfg, MODEL_IDS[1], rho=0.0
    )
    with torch.no_grad():
        embeddings = {
            "m1": m1.propagate(),
            "m2_rho0": m2_rho0.propagate(),
            "m2": m2.propagate(),
        }
    topk_change, score_effect, rank_boundary = _candidate_diagnostics(
        embeddings, prepared, cfg
    )
    condition_variation = _condition_diagnostics(prepared)
    report = {
        "topk_change": topk_change,
        "score_effect": score_effect,
        "rank_boundary": rank_boundary,
        "condition_variation": condition_variation,
        "checkpoints": {
            "m1": {
                "path": str(m1_checkpoint),
                "sha256": file_sha256(m1_checkpoint),
            },
            "m2": {
                "path": str(m2_checkpoint),
                "sha256": file_sha256(m2_checkpoint),
            },
        },
    }
    report["paths"] = _persist(report, cfg)
    print("\n===== 1) Top-K 추천목록 변경률 =====")
    print(topk_change.to_string(index=False))
    print("\n===== 2) 후보점수 실효 크기 =====")
    print(score_effect.to_string(index=False))
    print("\n===== 3) Top-K 경계 대비 직접 CLV 영향 =====")
    print(rank_boundary.to_string(index=False))
    print("\n===== 4) 시점별 historical CLV 변화 =====")
    print(condition_variation.to_string(index=False))
    print("\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print("No training starts automatically. Call run_dynamic_multianchor_diagnostic().")
