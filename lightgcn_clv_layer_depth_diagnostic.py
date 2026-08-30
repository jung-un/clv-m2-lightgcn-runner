"""Checkpoint-only LightGCN layer-depth diagnostic for Dunnhumby and H&M.

The diagnostic asks whether the fixed historical-CLV groups prefer different
amounts of collaborative propagation.  It does not train a model, select a
checkpoint, or construct the protected final test/holdout.  Every view uses
the same existing M1 layer-0 parameters and binary adjacency matrix:

* ``layer0``: E^(0)
* ``layer0_1_mean``: (E^(0) + E^(1)) / 2
* ``layer0_1_2_mean``: (E^(0) + E^(1) + E^(2)) / 3 (ordinary M1)

Changing the layer average only for this mechanism diagnostic is not an M2
result.  A CLV-conditioned aggregation may be implemented later only if the
same interpretable group pattern appears in both datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_gate_hm2y as hm2y
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_fixed_segment_error_diagnostic_hm2y as fixed_hm2y
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gatefree_lowdim_diagnostic as checkpoint_common
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m1-fixed-clv-layer-depth-diagnostic-v1"
VIEW_ORDER = ("layer0", "layer0_1_mean", "layer0_1_2_mean")
REFERENCE_VIEW = "layer0_1_2_mean"
GROUP_SPECS = (
    ("overall", "전체"),
    ("fixed_clv_segment", None),
    ("nv_quadrant", None),
    ("high_clv_composition", None),
)


@dataclass(frozen=True)
class LayerDepthDiagnosticConfig:
    dataset: str
    out_dir: str = ""
    baseline_result_dir: str = ""
    m1_checkpoint_dir: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 64
    max_k: int = 50


def configure_layer_depth_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> LayerDepthDiagnosticConfig:
    dataset = dataset.lower()
    if dataset not in {"dunnhumby", "hm"}:
        raise ValueError("dataset은 dunnhumby 또는 hm이어야 합니다")
    if dataset == "dunnhumby":
        defaults = gatefree.configure_gatefree_lowdim_run()
        values = {
            "dataset": dataset,
            "out_dir": (
                f"{v3.default_out_dir('dunnhumby')}"
                "_m1_fixed_clv_layer_depth_diagnostic_v1"
            ),
            "baseline_result_dir": defaults.baseline_result_dir,
            "eval_batch_size": 32,
        }
    else:
        values = {
            "dataset": dataset,
            "out_dir": (
                f"{v3.default_out_dir('hm')}"
                "_m1_fixed_clv_layer_depth_diagnostic_hm2y_v1"
            ),
            "m1_checkpoint_dir": v3.default_out_dir("hm"),
            "eval_batch_size": 256,
        }
    values.update(overrides)
    cfg = LayerDepthDiagnosticConfig(**values)
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    if dataset == "dunnhumby" and not (
        cfg.baseline_result_dir or cfg.m1_checkpoint
    ):
        raise ValueError("Dunnhumby M1 checkpoint 위치가 필요합니다")
    if dataset == "hm" and not (cfg.m1_checkpoint_dir or cfg.m1_checkpoint):
        raise ValueError("H&M M1 checkpoint 위치가 필요합니다")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size는 양수여야 합니다")
    if cfg.max_k != 50:
        raise ValueError("기존 @10·20·50 평가와 일치하도록 max_k=50만 허용합니다")
    return cfg


def preflight_summary(cfg: LayerDepthDiagnosticConfig) -> dict:
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
        "views": list(VIEW_ORDER),
        "reference_view": REFERENCE_VIEW,
        "fixed_clv_source": "training-window historical N×V proxy",
        "groups": [
            "overall",
            "fixed low/mid/high CLV",
            "N/V quadrants",
            "high-CLV N-dominant/balanced/V-dominant composition",
        ],
        "question": (
            "whether shallower collaborative propagation recovers M1 misses "
            "for fixed high-CLV users in both datasets"
        ),
        "reading_rule": (
            "depth conditioning is worth implementing only if a shallower "
            "view improves high-CLV Recall@10 and NDCG@10 versus ordinary "
            "M1, changes actual Top-10 sets, and the direction is coherent "
            "across Dunnhumby and H&M"
        ),
        "statistical_note": (
            "single-checkpoint descriptive mechanism diagnostic; no "
            "statistical significance claim"
        ),
        "out_dir": cfg.out_dir,
    }


def propagation_layers(model) -> list[torch.Tensor]:
    """Return E^(0)..E^(L) from the existing M1 without changing parameters."""
    n_layers = int(model.cfg["N_LAYERS"])
    if n_layers != 2:
        raise ValueError(f"이 진단은 기존 2층 M1만 허용합니다: {n_layers}")
    current = torch.cat([model.E_u.weight, model.E_i.weight], dim=0)
    layers = [current]
    for _ in range(n_layers):
        current = torch.sparse.mm(model.adj, current)
        layers.append(current)
    return layers


def aggregate_layer_views(
    model, layers: list[torch.Tensor]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if len(layers) != 3:
        raise ValueError("layer 0, 1, 2의 세 텐서가 필요합니다")
    node_views = {
        "layer0": layers[0],
        "layer0_1_mean": (layers[0] + layers[1]) / 2.0,
        "layer0_1_2_mean": (layers[0] + layers[1] + layers[2]) / 3.0,
    }
    return {
        name: (nodes[: model.n_users], nodes[model.n_users :])
        for name, nodes in node_views.items()
    }


@torch.no_grad()
def assert_full_view_parity(
    model,
    views: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    atol: float = 1e-6,
) -> float:
    # CUDA sparse.mm accumulates neighbouring rows with float32 atomics.  Two
    # mathematically identical passes can therefore differ by several ULPs;
    # 1e-6 accepts that numerical noise while still rejecting a changed view.
    expected_user, expected_item = model.propagate_pref()
    actual_user, actual_item = views[REFERENCE_VIEW]
    max_error = max(
        float((actual_user - expected_user).abs().max()),
        float((actual_item - expected_item).abs().max()),
    )
    if not torch.allclose(actual_user, expected_user, rtol=0.0, atol=atol):
        raise RuntimeError(f"M1 사용자 표현 parity 실패: max_error={max_error}")
    if not torch.allclose(actual_item, expected_item, rtol=0.0, atol=atol):
        raise RuntimeError(f"M1 상품 표현 parity 실패: max_error={max_error}")
    return max_error


def membership_for_evaluation_users(
    users: np.ndarray,
    *,
    fixed_segments: np.ndarray,
    high_compositions: np.ndarray,
    nv_quadrants: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a row-aligned membership frame; global IDs are never row indexes."""
    users = np.asarray(users, dtype=np.int64)
    fixed_segments = np.asarray(fixed_segments, dtype=object)
    high_compositions = np.asarray(high_compositions, dtype=object)
    if nv_quadrants is None:
        nv_quadrants = np.full(len(users), "계산불가", dtype=object)
    else:
        nv_quadrants = np.asarray(nv_quadrants, dtype=object)
    if not (len(users) == len(fixed_segments) == len(high_compositions) == len(nv_quadrants)):
        raise ValueError("평가 사용자와 그룹 배열의 길이가 다릅니다")
    return pd.DataFrame(
        {
            "user_idx": users,
            "fixed_clv_segment": fixed_segments,
            "nv_quadrant": nv_quadrants,
            "high_clv_composition": high_compositions,
        }
    )


def topk_overlap_rows(
    *,
    users: np.ndarray,
    reference_topk: np.ndarray,
    alternative_topk: np.ndarray,
    membership: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    users = np.asarray(users, dtype=np.int64)
    reference = np.asarray(reference_topk)[:, :k]
    alternative = np.asarray(alternative_topk)[:, :k]
    if reference.shape != alternative.shape or len(users) != len(reference):
        raise ValueError("Top-K 배열과 사용자 배열의 크기가 다릅니다")
    rows = []
    for user, ref, alt in zip(users, reference, alternative, strict=True):
        ref_set, alt_set = set(map(int, ref)), set(map(int, alt))
        union = ref_set | alt_set
        rows.append(
            {
                "user_idx": int(user),
                "topk_set_changed": float(ref_set != alt_set),
                "topk_order_changed": float(not np.array_equal(ref, alt)),
                "topk_jaccard": len(ref_set & alt_set) / len(union),
            }
        )
    return pd.DataFrame(rows).merge(
        membership, on="user_idx", how="left", validate="one_to_one"
    )


def _prepare_and_load(cfg: LayerDepthDiagnosticConfig):
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
    valid_clv = np.asarray(axes["clv_proxy"], dtype=np.float64)
    valid_clv = valid_clv[np.asarray(axes["valid_user"], dtype=bool)]
    low_edge, high_edge = prepared["base_cfg"]["SEG_EDGES"]
    thresholds = tuple(
        float(value)
        for value in np.quantile(valid_clv, [low_edge, high_edge])
    )
    return fixed.build_user_value_groups(
        axes,
        clv_thresholds=thresholds,
        evaluation_users=prepared["cache"].users,
    )


def _group_masks(frame: pd.DataFrame):
    yield "overall", "전체", np.ones(len(frame), dtype=bool)
    orders = {
        "fixed_clv_segment": fixed.SEGMENT_ORDER,
        "nv_quadrant": fixed.NV_QUADRANT_ORDER,
        "high_clv_composition": fixed.HIGH_CLV_COMPOSITION_ORDER,
    }
    for column, order in orders.items():
        values = frame[column].to_numpy()
        for group in order:
            mask = values == group
            if mask.any():
                yield column, group, mask


def _metric_summary(per_user_views: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_user_views if "@" in column]
    rows = []
    for view in VIEW_ORDER:
        selected = per_user_views[per_user_views.view.eq(view)].reset_index(drop=True)
        for group_type, group, mask in _group_masks(selected):
            group_frame = selected.loc[mask]
            row = {
                "view": view,
                "group_type": group_type,
                "group": group,
                "n_users": len(group_frame),
                "mean_truth_items": group_frame.truth_item_count.mean(),
                "mean_q_n": group_frame.q_n.mean(),
                "mean_q_v": group_frame.q_v.mean(),
                "mean_historical_clv_proxy": group_frame.historical_clv_proxy.mean(),
            }
            row.update({column: group_frame[column].mean() for column in metric_columns})
            rows.append(row)
    return pd.DataFrame(rows)


def _overlap_summary(
    users: np.ndarray,
    top50_by_view: dict[str, np.ndarray],
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    reference = top50_by_view[REFERENCE_VIEW]
    for view in VIEW_ORDER:
        if view == REFERENCE_VIEW:
            continue
        overlap10 = topk_overlap_rows(
            users=users,
            reference_topk=reference,
            alternative_topk=top50_by_view[view],
            membership=membership,
            k=10,
        )
        overlap50 = topk_overlap_rows(
            users=users,
            reference_topk=reference,
            alternative_topk=top50_by_view[view],
            membership=membership,
            k=50,
        )
        for group_type, group, mask in _group_masks(overlap10):
            a, b = overlap10.loc[mask], overlap50.loc[mask]
            rows.append(
                {
                    "view": view,
                    "reference_view": REFERENCE_VIEW,
                    "group_type": group_type,
                    "group": group,
                    "n_users": len(a),
                    "top10_set_changed_user_share": a.topk_set_changed.mean(),
                    "top10_order_changed_user_share": a.topk_order_changed.mean(),
                    "top10_mean_jaccard": a.topk_jaccard.mean(),
                    "top50_set_changed_user_share": b.topk_set_changed.mean(),
                    "top50_order_changed_user_share": b.topk_order_changed.mean(),
                    "top50_mean_jaccard": b.topk_jaccard.mean(),
                }
            )
    return pd.DataFrame(rows)


def _comparison(summary: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["group_type", "group"]
    reference = summary[summary.view.eq(REFERENCE_VIEW)].set_index(identifiers)
    metric_columns = [column for column in summary if "@" in column]
    rows = []
    for _, row in summary[~summary.view.eq(REFERENCE_VIEW)].iterrows():
        key = (row.group_type, row.group)
        for metric in metric_columns:
            reference_value = float(reference.at[key, metric])
            model_value = float(row[metric])
            rows.append(
                {
                    "view": row.view,
                    "reference_view": REFERENCE_VIEW,
                    "group_type": row.group_type,
                    "group": row.group,
                    "metric": metric,
                    "reference_value": reference_value,
                    "view_value": model_value,
                    "absolute_delta": model_value - reference_value,
                    "relative_change_pct": (
                        100.0 * (model_value / reference_value - 1.0)
                        if reference_value != 0.0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).merge(
        overlap,
        on=["view", "reference_view", "group_type", "group"],
        how="left",
        validate="many_to_one",
    )


def _result_paths(cfg: LayerDepthDiagnosticConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m1_layer_depth_{cfg.dataset}"
    return {
        "view_metrics_csv": root / f"{stem}_view_metrics.csv",
        "comparison_csv": root / f"{stem}_comparison.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_layer_depth_diagnostic(
    cfg: LayerDepthDiagnosticConfig | None = None,
) -> dict:
    cfg = cfg or configure_layer_depth_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = _prepare_and_load(cfg)
    model.eval()
    layers = propagation_layers(model)
    views = aggregate_layer_views(model, layers)
    parity_error = assert_full_view_parity(model, views)
    membership_all, thresholds = _membership(prepared, axes)
    users = np.asarray(prepared["cache"].users, dtype=np.int64)
    by_user = membership_all.set_index("user_idx")
    membership = membership_for_evaluation_users(
        users,
        fixed_segments=by_user.loc[users, "fixed_clv_segment"].to_numpy(),
        nv_quadrants=by_user.loc[users, "nv_quadrant"].to_numpy(),
        high_compositions=by_user.loc[users, "high_clv_composition"].to_numpy(),
    ).merge(
        membership_all[
            [
                "user_idx",
                "n_behavior_score",
                "v_behavior_score",
                "historical_clv_proxy",
                "q_n",
                "q_v",
                "q_n_minus_q_v",
            ]
        ],
        on="user_idx",
        how="left",
        validate="one_to_one",
    )

    top50_by_view: dict[str, np.ndarray] = {}
    per_user_frames = []
    for view in VIEW_ORDER:
        user_embedding, item_embedding = views[view]
        ranked_users, top50 = item_fit._masked_topk(
            user_embedding,
            item_embedding,
            prepared,
            max_k=cfg.max_k,
            batch_size=cfg.eval_batch_size,
        )
        if not np.array_equal(ranked_users, users):
            raise RuntimeError("레이어별 평가 사용자 순서가 달라졌습니다")
        top50_by_view[view] = top50
        per_user = fixed._per_user_metrics(
            users=users, top50=top50, prepared=prepared
        ).merge(membership, on="user_idx", how="left", validate="one_to_one")
        per_user.insert(0, "view", view)
        per_user_frames.append(per_user)
        print(f"  [{cfg.dataset}] {view} Top-{cfg.max_k} 평가 완료")

    per_user_views = pd.concat(per_user_frames, ignore_index=True)
    summary = _metric_summary(per_user_views)
    overlap = _overlap_summary(users, top50_by_view, membership)
    comparison = _comparison(summary, overlap)
    paths = _result_paths(cfg)
    test10._atomic_csv(paths["view_metrics_csv"], summary)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["per_user_csv"], per_user_views)
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
        "full_view_max_parity_error": parity_error,
        "view_metrics": summary.to_dict("records"),
        "comparison": comparison.to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "interpretation_limits": [
            "checkpoint-only descriptive mechanism diagnostic",
            "no training, checkpoint selection, or hyperparameter selection",
            "ordinary M1 is layer0_1_2_mean",
            "no statistical significance claim",
            "historical CLV proxy is not future or incremental CLV",
            "price/purchase-amount weighted hit is not actual incremental revenue",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}
