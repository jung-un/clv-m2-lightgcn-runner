"""Checkpoint-only diagnostics for the independent-axis dropout M2 screen.

This module never trains or selects a checkpoint.  It reloads the fixed seed-42
historical-development M1 and M2 checkpoints, then measures which propagated
block (ID, activity N, or transaction-value V) changes held-out ranking.
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
import lightgcn_clv_gatefree_lowdim_independent_dropout as dropout
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-independent-axis-dropout-checkpoint-diagnostic-v1"
VIEW_MODES = ("m1_64", "id_only", "id_n", "id_v", "full")
M2_VIEW_MODES = ("id_only", "id_n", "id_v", "full")
SEGMENT_ORDER = ("전체", "저CLV", "중CLV", "고CLV")
CORE_METRICS = tuple(
    metric
    for cutoff in (10, 20, 50)
    for metric in (
        f"recall@{cutoff}",
        f"ndcg@{cutoff}",
        f"price_purchase_amount_weighted_hit@{cutoff}",
    )
)


@dataclass(frozen=True)
class IndependentDropoutDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m2_checkpoint: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 32


def configure_independent_dropout_diagnostic(
    **overrides,
) -> IndependentDropoutDiagnosticConfig:
    defaults = dropout.configure_independent_dropout_run()
    values = {
        "out_dir": defaults.out_dir,
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = IndependentDropoutDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: IndependentDropoutDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "checkpoint_selection": False,
        "split": "historical_development_days_684_690",
        "views": list(VIEW_MODES),
        "segments": list(SEGMENT_ORDER),
        "questions": [
            "which axis changes overall and segment ranking metrics",
            "which axis moves truth items into or out of Top-10/20/50",
            "how large each axis score is relative to the ID score",
        ],
        "statistical_note": "descriptive seed-42 checkpoint diagnostic; no significance claim",
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _load_m2(prepared: dict, runner_cfg, cfg: IndependentDropoutDiagnosticConfig):
    if cfg.m2_checkpoint:
        checkpoint = Path(cfg.m2_checkpoint)
        record = {}
    else:
        checkpoint, record = common._checkpoint_record(
            Path(cfg.out_dir), dropout.MODEL_ID
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
        "hidden_dim",
        "axis_budget",
        "axis_keep_probability",
        "n_layers",
        "input_days",
    )
    mismatch = {
        key: {"expected": expected[key], "actual": recorded.get(key)}
        for key in identity_keys
        if recorded.get(key) != expected[key]
    }
    if payload.get("model_id") != dropout.MODEL_ID:
        mismatch["model_id"] = {
            "expected": dropout.MODEL_ID,
            "actual": payload.get("model_id"),
        }
    if payload.get("input_hash") != prepared["input_hash"]:
        mismatch["input_hash"] = {
            "expected": prepared["input_hash"],
            "actual": payload.get("input_hash"),
        }
    if mismatch:
        raise RuntimeError(f"M2 checkpoint identity mismatch: {mismatch}")
    model, _ = dropout._build_model(prepared, runner_cfg)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


def _flat_metrics(user: torch.Tensor, item: torch.Tensor, prepared: dict) -> dict:
    metrics, _ = moe._flat_evaluation(
        common._FixedEmbeddingView(user, item),
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    return test10._public_metrics(metrics)


def _view_metrics(
    *,
    m1_user: torch.Tensor,
    m1_item: torch.Tensor,
    views: dict[str, tuple[torch.Tensor, torch.Tensor]],
    prepared: dict,
) -> pd.DataFrame:
    rows = [{"view": "m1_64", **_flat_metrics(m1_user, m1_item, prepared)}]
    for name in M2_VIEW_MODES:
        rows.append({"view": name, **_flat_metrics(*views[name], prepared)})
    return pd.DataFrame(rows)


def _metric_column(frame: pd.DataFrame, segment: str, metric: str) -> str | None:
    column = metric if segment == "전체" else f"{segment}_{metric}"
    if column in frame.columns:
        return column
    # Compatibility with checkpoints created before the public metric rename.
    legacy = column.replace("price_purchase_amount_weighted_hit", "revenue")
    return legacy if legacy in frame.columns else None


def axis_effect_table(view_metrics: pd.DataFrame) -> pd.DataFrame:
    """Long table of M1 and N/V contribution, using ID-only as axis reference."""
    indexed = view_metrics.set_index("view")
    required = set(VIEW_MODES)
    if not required.issubset(indexed.index):
        raise ValueError(f"필요 view가 없습니다: {sorted(required - set(indexed.index))}")
    rows = []
    for segment in SEGMENT_ORDER:
        for metric in CORE_METRICS:
            column = _metric_column(view_metrics, segment, metric)
            if column is None:
                continue
            values = {name: float(indexed.at[name, column]) for name in VIEW_MODES}
            row = {"segment": segment, "metric": metric, **values}
            for name in ("id_n", "id_v", "full"):
                delta = values[name] - values["id_only"]
                row[f"{name}_delta_vs_id_only"] = delta
                row[f"{name}_relative_pct_vs_id_only"] = (
                    100.0 * delta / values["id_only"]
                    if values["id_only"] != 0
                    else np.nan
                )
            row["full_delta_vs_m1"] = values["full"] - values["m1_64"]
            row["full_relative_pct_vs_m1"] = (
                100.0 * row["full_delta_vs_m1"] / values["m1_64"]
                if values["m1_64"] != 0
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def rank_flow_summary(
    transition: pd.DataFrame,
    *,
    reference: str,
    view: str,
) -> pd.DataFrame:
    """Summarize truth items entering/leaving each Top-K from a transition table."""
    order = {"1-10": 10, "11-20": 20, "21-50": 50, ">50": 51}
    rows = []
    for segment in common.SEGMENT_ORDER:
        subset = transition[transition["segment"] == segment]
        total = int(subset["truth_item_count"].sum())
        for cutoff in (10, 20, 50):
            reference_in = subset["reference_bucket"].map(order).le(cutoff)
            model_in = subset["model_bucket"].map(order).le(cutoff)
            entered = int(subset.loc[~reference_in & model_in, "truth_item_count"].sum())
            exited = int(subset.loc[reference_in & ~model_in, "truth_item_count"].sum())
            retained = int(subset.loc[reference_in & model_in, "truth_item_count"].sum())
            rows.append(
                {
                    "reference": reference,
                    "view": view,
                    "segment": segment,
                    "cutoff": cutoff,
                    "truth_item_count": total,
                    "entered_topk": entered,
                    "exited_topk": exited,
                    "net_truth_items": entered - exited,
                    "retained_topk": retained,
                }
            )
    return pd.DataFrame(rows)


def _rank_reports(
    *,
    ranks: dict[str, dict[int, dict[int, int]]],
    prepared: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("m1_64", "full"),
        ("id_only", "id_n"),
        ("id_only", "id_v"),
        ("id_only", "full"),
    )
    transitions, flows = [], []
    for reference, view in comparisons:
        table = common.rank_transition_table(
            users=prepared["cache"].users,
            segments=prepared["cache"].seg,
            truth=prepared["cache"].gt,
            reference_ranks=ranks[reference],
            model_ranks=ranks[view],
        )
        table.insert(0, "view", view)
        table.insert(0, "reference", reference)
        transitions.append(table)
        flows.append(rank_flow_summary(table, reference=reference, view=view))
    return pd.concat(transitions, ignore_index=True), pd.concat(flows, ignore_index=True)


def _persist(report: dict, cfg: IndependentDropoutDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "view_metrics_csv": root / "m2_independent_dropout_axis_view_metrics.csv",
        "axis_effect_csv": root / "m2_independent_dropout_axis_effect_by_segment.csv",
        "rank_transition_csv": root / "m2_independent_dropout_rank_transition.csv",
        "rank_flow_csv": root / "m2_independent_dropout_rank_flow_summary.csv",
        "score_strength_csv": root / "m2_independent_dropout_score_strength.csv",
        "json": root / "m2_independent_dropout_checkpoint_diagnostic.json",
    }
    for key, report_key in (
        ("view_metrics_csv", "view_metrics"),
        ("axis_effect_csv", "axis_effect"),
        ("rank_transition_csv", "rank_transition"),
        ("rank_flow_csv", "rank_flow"),
        ("score_strength_csv", "score_strength"),
    ):
        test10._atomic_csv(paths[key], report[report_key])
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoints only; no training or checkpoint selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoints": report["checkpoints"],
        "view_metrics": report["view_metrics"].to_dict("records"),
        "axis_effect": report["axis_effect"].to_dict("records"),
        "rank_transition": report["rank_transition"].to_dict("records"),
        "rank_flow": report["rank_flow"].to_dict("records"),
        "score_strength": report["score_strength"].to_dict("records"),
        "interpretation_limits": [
            "descriptive mechanism diagnostic only",
            "seed 42 only; no significance claim",
            "does not train or select another model",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_independent_dropout_diagnostic(
    cfg: IndependentDropoutDiagnosticConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or configure_independent_dropout_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = dropout.configure_independent_dropout_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = dropout._prepare(runner_cfg)
    m2_model, m2_checkpoint, _ = _load_m2(prepared, runner_cfg, cfg)
    m1_model, m1_checkpoint, _ = common._load_m1(prepared, cfg)
    with torch.no_grad():
        m2_user, m2_item = m2_model.propagate()
        m1_user, m1_item = m1_model.propagate_pref()
    views = common.axis_views(
        m2_user,
        m2_item,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
    )
    view_metrics = _view_metrics(
        m1_user=m1_user,
        m1_item=m1_item,
        views=views,
        prepared=prepared,
    )
    axis_effect = axis_effect_table(view_metrics)
    ranks = {
        "m1_64": common._top50_ranks(
            m1_user, m1_item, prepared, batch_size=cfg.eval_batch_size
        )
    }
    for name in M2_VIEW_MODES:
        ranks[name] = common._top50_ranks(
            *views[name], prepared, batch_size=cfg.eval_batch_size
        )
    rank_transition, rank_flow = _rank_reports(ranks=ranks, prepared=prepared)
    score_strength = common._score_strength(
        m2_user,
        m2_item,
        prepared,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
        batch_size=cfg.eval_batch_size,
    )
    report = {
        "view_metrics": view_metrics,
        "axis_effect": axis_effect,
        "rank_transition": rank_transition,
        "rank_flow": rank_flow,
        "score_strength": score_strength,
        "checkpoints": {
            "m1": {"path": str(m1_checkpoint), "sha256": file_sha256(m1_checkpoint)},
            "m2": {"path": str(m2_checkpoint), "sha256": file_sha256(m2_checkpoint)},
        },
    }
    report["paths"] = _persist(report, cfg)
    print("\n===== 1. ID/N/V 블록별 전체·CLV 구간 효과 =====")
    print(axis_effect.to_string(index=False))
    print("\n===== 2. 각 축이 Top-K 정답을 넣고 뺀 개수 =====")
    print(rank_flow.to_string(index=False))
    print("\n===== 3. 전체 절대지표 =====")
    print(view_metrics.to_string(index=False))
    print("\n===== 4. 실제 후보점수 영향력 =====")
    print(score_strength.to_string(index=False))
    print("\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print("No training is started automatically. Call run_independent_dropout_diagnostic().")
