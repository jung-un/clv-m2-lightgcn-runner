"""H&M 2-year M1 errors by fixed historical-CLV and N/V composition.

This module is evaluation-only.  It loads the existing seed-42 H&M M1
checkpoint, keeps test and holdout closed, and analyses the existing validation
ranking.  Historical customer value is computed from the training window only:

    N = number of purchase occasions, where an H&M occasion is (user, date)
    V = mean purchase amount per occasion
    historical CLV proxy = N * V = observed training-window purchase amount

The proxy is used only to form descriptive user groups.  No model is trained
and no checkpoint or hyperparameter is selected here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_dual_axis_model import fixed_percentile_ranks
from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_gate_hm2y as hm2y
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_fixed_segment_error_diagnostic as common
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m1-fixed-clv-nv-segment-error-diagnostic-hm2y-v1"


@dataclass(frozen=True)
class Hm2YFixedSegmentErrorDiagnosticConfig:
    out_dir: str = ""
    m1_checkpoint_dir: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 256
    top_examples: int = 20


def configure_hm2y_fixed_segment_error_diagnostic(
    **overrides,
) -> Hm2YFixedSegmentErrorDiagnosticConfig:
    values = {
        "out_dir": (
            f"{v3.default_out_dir('hm')}"
            "_m1_fixed_clv_nv_segment_error_diagnostic_hm2y_v1"
        ),
        "m1_checkpoint_dir": v3.default_out_dir("hm"),
    }
    values.update(overrides)
    cfg = Hm2YFixedSegmentErrorDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.m1_checkpoint_dir:
        raise ValueError("out_dir와 m1_checkpoint_dir가 필요합니다")
    if cfg.eval_batch_size <= 0 or cfg.top_examples <= 0:
        raise ValueError("배치 크기와 산출 예시 수는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: Hm2YFixedSegmentErrorDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full_history_about_2_years",
        "training": False,
        "checkpoint_selection": False,
        "model": "existing seed-42 H&M M1@64 checkpoint",
        "split": "existing_hm2y_validation",
        "test_executed": False,
        "holdout_executed": False,
        "new_item_task": True,
        "fixed_clv_source": (
            "training-window purchase occasions only; H&M occasion=(customer,date)"
        ),
        "fixed_clv_definition": {
            "N": "number of distinct customer-date purchase occasions",
            "V": "mean summed purchase amount per customer-date occasion",
            "historical_clv_proxy": "N*V (training-window observed purchase amount)",
        },
        "segments": list(common.SEGMENT_ORDER),
        "nv_quadrants": list(common.NV_QUADRANT_ORDER),
        "high_clv_compositions": list(common.HIGH_CLV_COMPOSITION_ORDER),
        "item_traits": {
            "category": "H&M product_group_name",
            "price": "training-window item mean-price percentile",
            "popularity": "number of distinct training customers",
            "repeat": (
                "exact-article repeat-purchase share; descriptive only because "
                "exact repeats are sparse in fashion"
            ),
            "personal_fit": (
                "own-history product-group overlap and M1 item-embedding cosine"
            ),
        },
        "interpretation": (
            "descriptive validation diagnostic only; compare whether the same "
            "fixed CLV/N/V logic has similar or different error profiles in H&M"
        ),
        "statistical_note": "single-seed descriptive diagnostic; no significance claim",
        "out_dir": cfg.out_dir,
        "m1_checkpoint_dir": cfg.m1_checkpoint_dir,
    }


def build_purchase_occasion_axes(train: pd.DataFrame, n_users: int) -> dict:
    """Return fixed train-only N, V and N*V using H&M customer-date occasions."""
    required = {"u_idx", "t", "v"}
    missing = required.difference(train.columns)
    if missing:
        raise KeyError(f"H&M 구매기회 계산 열이 없습니다: {sorted(missing)}")
    occasion = (
        train.groupby(["u_idx", "t"], sort=False)["v"]
        .sum()
        .rename("occasion_amount")
        .reset_index()
    )
    by_user = occasion.groupby("u_idx", sort=False).occasion_amount.agg(
        ["size", "mean"]
    )
    n_score = np.zeros(n_users, np.float32)
    v_score = np.zeros(n_users, np.float32)
    valid = np.zeros(n_users, bool)
    ids = by_user.index.to_numpy(np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= n_users):
        raise ValueError("구매기회 사용자 ID가 n_users 범위를 벗어났습니다")
    n_score[ids] = by_user["size"].to_numpy(np.float32)
    v_score[ids] = by_user["mean"].to_numpy(np.float32)
    valid[ids] = True
    clv_proxy = n_score * v_score
    q_n, q_v = fixed_percentile_ranks(n_score, v_score, valid)
    return {
        "n_behavior_score": n_score,
        "v_behavior_score": v_score,
        "clv_proxy": clv_proxy,
        "q_n": q_n,
        "q_v": q_v,
        "valid_user": valid,
        "activity_valid": valid.copy(),
        "value_valid": valid.copy(),
    }


def _checkpoint_path(prepared: dict, cfg: Hm2YFixedSegmentErrorDiagnosticConfig) -> Path:
    if cfg.m1_checkpoint:
        return Path(cfg.m1_checkpoint)
    base_cfg = prepared["base_cfg"]
    return Path(base_cfg["OUT_DIR"]) / (
        "ckpt_pref_only_hm_s42_"
        f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', 42)}.pt"
    )


def _load_existing_m1(prepared: dict, cfg: Hm2YFixedSegmentErrorDiagnosticConfig):
    checkpoint = _checkpoint_path(prepared, cfg)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "기존 H&M 2년 M1 checkpoint가 없습니다. 진단은 학습을 시작하지 않습니다: "
            f"{checkpoint}"
        )
    blob = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    if "state" not in blob:
        raise RuntimeError(f"M1 checkpoint에 state가 없습니다: {checkpoint}")
    model = v3.build_model(
        prepared["data"],
        prepared["data"]["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        prepared["base_cfg"],
    )
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, checkpoint


def attach_history_relations_from_graph(
    occurrences: pd.DataFrame,
    *,
    csr_ptr: np.ndarray,
    csr_items: np.ndarray,
    item_traits: pd.DataFrame,
    item_embedding: np.ndarray,
) -> pd.DataFrame:
    """Attach own-history relations without an n_users x dim dense H&M array."""
    output = occurrences.copy()
    embedding = np.asarray(item_embedding, dtype=np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    unit_items = np.divide(
        embedding, norms, out=np.zeros_like(embedding), where=norms > 0
    )
    categories = (
        item_traits.set_index("item_idx")["category"]
        .reindex(np.arange(len(embedding)))
        .fillna("UNKNOWN")
        .astype(str)
        .to_numpy()
    )
    users = np.sort(output.user_idx.unique().astype(np.int64))
    centroids = np.zeros((len(users), embedding.shape[1]), np.float32)
    history_categories: dict[int, set[str]] = {}
    for position, user in enumerate(users):
        left, right = int(csr_ptr[user]), int(csr_ptr[user + 1])
        items = np.asarray(csr_items[left:right], dtype=np.int64)
        if len(items):
            centroids[position] = unit_items[items].mean(axis=0)
            history_categories[int(user)] = set(categories[items])
        else:
            history_categories[int(user)] = set()
    centroid_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = np.divide(
        centroids,
        centroid_norm,
        out=np.zeros_like(centroids),
        where=centroid_norm > 0,
    )
    positions = np.searchsorted(users, output.user_idx.to_numpy(np.int64))
    item_indices = output.item_idx.to_numpy(np.int64)
    output["history_category_overlap"] = [
        float(str(category) in history_categories[int(user)])
        for user, category in zip(
            output.user_idx, output.category, strict=True
        )
    ]
    output["history_embedding_cosine"] = (
        centroids[positions] * unit_items[item_indices]
    ).sum(axis=1)
    return output


def _persist(report: dict, cfg: Hm2YFixedSegmentErrorDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        "segment_population": report["segment_population"],
        "segment_metrics": report["segment_metrics"],
        "nv_quadrant_population": report["nv_quadrant_population"],
        "nv_quadrant_metrics": report["nv_quadrant_metrics"],
        "nv_quadrant_item_role_summary": report["nv_quadrant_item_role_summary"],
        "nv_quadrant_contrasts": report["nv_quadrant_contrasts"],
        "nv_quadrant_category_summary": report["nv_quadrant_category_summary"],
        "nv_quadrant_examples": report["nv_quadrant_examples"],
        "high_clv_composition_population": report["high_clv_composition_population"],
        "high_clv_composition_metrics": report["high_clv_composition_metrics"],
        "high_clv_composition_item_role_summary": report[
            "high_clv_composition_item_role_summary"
        ],
        "high_clv_composition_contrasts": report[
            "high_clv_composition_contrasts"
        ],
        "high_clv_composition_category_summary": report[
            "high_clv_composition_category_summary"
        ],
        "high_clv_composition_examples": report["high_clv_composition_examples"],
        "item_role_summary": report["item_role_summary"],
        "miss_false_positive_contrasts": report["contrasts"],
        "category_summary": report["category_summary"],
        "examples": report["examples"],
    }
    paths = {
        name: root / f"m1_hm2y_fixed_clv_nv_{name}.csv" for name in frames
    }
    paths["json"] = root / "m1_hm2y_fixed_clv_nv_error_diagnostic.json"
    for name, frame in frames.items():
        test10._atomic_csv(paths[name], frame)
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoint": report["checkpoint"],
        "data_stats": report["data_stats"],
        "group_thresholds": report["group_thresholds"],
        **{name: frame.to_dict("records") for name, frame in frames.items()},
        "result_paths": {name: str(path) for name, path in paths.items()},
        "reading_rule": {
            "cross_dataset_question": (
                "whether H&M error profiles vary by fixed CLV level and N/V "
                "composition in the same direction as Dunnhumby"
            ),
            "supports_shared_conditioning_if": (
                "the relevant segment/composition differences appear in both datasets"
            ),
            "otherwise": (
                "do not encode a Dunnhumby-specific repeat-product relation as the "
                "shared M2 mechanism"
            ),
        },
        "interpretation_limits": [
            "validation-only descriptive diagnostic",
            "no training or checkpoint selection",
            "no statistical significance claim",
            "historical CLV proxy is not future or incremental CLV",
            "price/purchase-amount weighted hit is not actual incremental revenue",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_hm2y_fixed_segment_error_diagnostic(
    cfg: Hm2YFixedSegmentErrorDiagnosticConfig | None = None,
) -> dict:
    cfg = cfg or configure_hm2y_fixed_segment_error_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = hm2y.configure_axis_specific_gate_hm2y_run(
        out_dir=cfg.out_dir,
        m1_checkpoint_dir=cfg.m1_checkpoint_dir,
    )
    prepared = joint._prepare(runner_cfg)
    model, checkpoint = _load_existing_m1(prepared, cfg)
    with torch.no_grad():
        user_embedding, item_embedding, *_ = model.embeddings(need_value=False)
    users, top50 = item_fit._masked_topk(
        user_embedding,
        item_embedding,
        prepared,
        max_k=50,
        batch_size=cfg.eval_batch_size,
    )

    axes = build_purchase_occasion_axes(
        prepared["data"]["train"], prepared["data"]["n_users"]
    )
    valid_clv = axes["clv_proxy"][axes["valid_user"]]
    low_edge, high_edge = prepared["base_cfg"]["SEG_EDGES"]
    clv_thresholds = tuple(
        float(value)
        for value in np.quantile(valid_clv, [low_edge, high_edge])
    )
    membership, thresholds = common.build_user_value_groups(
        axes,
        clv_thresholds=clv_thresholds,
        evaluation_users=users,
    )
    by_user = membership.set_index("user_idx")
    segments = by_user.loc[users, "fixed_clv_segment"].to_numpy()
    occurrences = common.item_role_occurrences(
        users=users,
        segments=segments,
        truth=prepared["cache"].gt,
        top50=top50,
        truth_amount=prepared["cache"].rev,
    )
    item_traits = common._raw_item_traits(
        prepared["data"]["train"], prepared["data"]["n_items"]
    )
    occurrences = occurrences.merge(item_traits, on="item_idx", how="left")
    occurrences = attach_history_relations_from_graph(
        occurrences,
        csr_ptr=prepared["data"]["csr_ptr"],
        csr_items=prepared["data"]["csr_items"],
        item_traits=item_traits,
        item_embedding=item_embedding.detach().cpu().numpy(),
    )
    occurrences = occurrences.merge(
        membership[["user_idx", "nv_quadrant", "high_clv_composition"]],
        on="user_idx",
        how="left",
        validate="many_to_one",
    )
    per_user = common._per_user_metrics(
        users=users, top50=top50, prepared=prepared
    )
    segment_population = common.summarize_user_groups(
        membership,
        group_column="fixed_clv_segment",
        group_order=common.SEGMENT_ORDER,
    )
    segment_metrics = common.summarize_metrics_by_group(
        per_user,
        membership,
        group_column="fixed_clv_segment",
        group_order=common.SEGMENT_ORDER,
    )
    nv_population = common.summarize_user_groups(
        membership,
        group_column="nv_quadrant",
        group_order=common.NV_QUADRANT_ORDER,
    )
    nv_metrics = common.summarize_metrics_by_group(
        per_user,
        membership,
        group_column="nv_quadrant",
        group_order=common.NV_QUADRANT_ORDER,
    )
    high_population = common.summarize_user_groups(
        membership,
        group_column="high_clv_composition",
        group_order=common.HIGH_CLV_COMPOSITION_ORDER,
    )
    high_metrics = common.summarize_metrics_by_group(
        per_user,
        membership,
        group_column="high_clv_composition",
        group_order=common.HIGH_CLV_COMPOSITION_ORDER,
    )

    item_role_summary = common.summarize_segment_item_roles(occurrences)
    contrasts = common.miss_false_positive_contrasts(item_role_summary)
    nv_occurrences = occurrences[
        occurrences.nv_quadrant.isin(common.NV_QUADRANT_ORDER)
    ]
    high_occurrences = occurrences[
        occurrences.high_clv_composition.isin(common.HIGH_CLV_COMPOSITION_ORDER)
    ]
    nv_item_summary = common.summarize_item_roles_by(
        nv_occurrences, group_column="nv_quadrant"
    )
    high_item_summary = common.summarize_item_roles_by(
        high_occurrences, group_column="high_clv_composition"
    )
    report = {
        "segment_population": segment_population,
        "segment_metrics": segment_metrics,
        "nv_quadrant_population": nv_population,
        "nv_quadrant_metrics": nv_metrics,
        "nv_quadrant_item_role_summary": nv_item_summary,
        "nv_quadrant_contrasts": common.contrasts_by_group(
            nv_item_summary,
            group_column="nv_quadrant",
            group_order=common.NV_QUADRANT_ORDER,
        ),
        "nv_quadrant_category_summary": common._category_summary(
            nv_occurrences, cfg.top_examples, group_column="nv_quadrant"
        ),
        "nv_quadrant_examples": common._examples(
            nv_occurrences, cfg.top_examples, group_column="nv_quadrant"
        ),
        "high_clv_composition_population": high_population,
        "high_clv_composition_metrics": high_metrics,
        "high_clv_composition_item_role_summary": high_item_summary,
        "high_clv_composition_contrasts": common.contrasts_by_group(
            high_item_summary,
            group_column="high_clv_composition",
            group_order=common.HIGH_CLV_COMPOSITION_ORDER,
        ),
        "high_clv_composition_category_summary": common._category_summary(
            high_occurrences,
            cfg.top_examples,
            group_column="high_clv_composition",
        ),
        "high_clv_composition_examples": common._examples(
            high_occurrences,
            cfg.top_examples,
            group_column="high_clv_composition",
        ),
        "item_role_summary": item_role_summary,
        "contrasts": contrasts,
        "category_summary": common._category_summary(
            occurrences, cfg.top_examples
        ),
        "examples": common._examples(occurrences, cfg.top_examples),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "data_stats": prepared["data"].get("data_stats", {}),
        "group_thresholds": thresholds,
    }
    report["paths"] = _persist(report, cfg)

    print("\n===== 1) H&M 2년 고정 CLV 구간별 M1 성과 =====")
    print(segment_metrics.to_string(index=False))
    print("\n===== 2) H&M 2년 전체 고객 N/V 4유형 성과 =====")
    print(nv_population.to_string(index=False))
    print(nv_metrics.to_string(index=False))
    print("\n===== 3) H&M 2년 고CLV 내부 N/V 구성별 성과 =====")
    print(high_population.to_string(index=False))
    print(high_metrics.to_string(index=False))
    print("\n===== 4) H&M 정답 누락 - Top-10 오추천 격차 =====")
    print(contrasts.to_string(index=False))
    print("\n===== 5) H&M 고CLV 내부 정답 누락 - Top-10 오추천 격차 =====")
    print(report["high_clv_composition_contrasts"].to_string(index=False))
    print("\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_hm2y_fixed_segment_error_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
