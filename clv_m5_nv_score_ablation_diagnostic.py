"""Read-only N/V score-component audit of one selected, jointly trained M5.

The ID-only score here is an inference-time diagnostic of the *same M5*
checkpoint. It is neither M1 nor a deployable/retrained ablation arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m4_m5_top10_movement_diagnostic as movement


VERSION = "m5-selected-nv-score-component-development-v1"
TOP_K = 50
KS = (10, 20, 50)
COMPONENTS = ("m5_full", "m5_id_component_only")


@torch.no_grad()
def rank_and_explain(model, prepared: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """Rank with the original M5 score and with its N/V summands omitted."""
    v3 = movement.screen.base.v3
    cache, data = prepared["cache"], prepared["data"]
    users = np.asarray(cache.users, dtype=np.int64)
    model.eval()
    user, item, _, _ = model.embeddings()
    id_dim, axis_dim = model.id_dim, model.axis_dim
    if (user.shape != (data["n_users"], id_dim + 2 * axis_dim)
            or item.shape != (data["n_items"], id_dim + 2 * axis_dim)
            or data["n_items"] < TOP_K):
        raise ValueError("M5 expression dimensions/catalog do not match the source protocol")
    result_full = np.empty((len(users), TOP_K), dtype=np.int64)
    result_id = np.empty_like(result_full)
    candidate_rows = []
    max_reconstruction_error = 0.0
    batch_size = min(int(prepared["base_cfg"]["EVAL_BATCH"]), 256)
    if batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        ut = torch.as_tensor(batch, dtype=torch.long, device=v3.DEVICE)
        # The full matrix multiplication matches the original evaluation path.
        full_scores = user[ut] @ item.T
        id_scores = user[ut, :id_dim] @ item[:, :id_dim].T
        for row, u in enumerate(batch):
            begin, end = data["csr_ptr"][u], data["csr_ptr"][u + 1]
            if end > begin:
                seen = data["csr_items"][begin:end]
                full_scores[row, seen] = -1e9
                id_scores[row, seen] = -1e9
        full_top = full_scores.topk(TOP_K, dim=1).indices.cpu().numpy()
        id_top = id_scores.topk(TOP_K, dim=1).indices.cpu().numpy()
        result_full[start:start + len(batch)] = full_top
        result_id[start:start + len(batch)] = id_top
        for row, u in enumerate(batch):
            entered = set(full_top[row, :10]) - set(id_top[row, :10])
            exited = set(id_top[row, :10]) - set(full_top[row, :10])
            if not entered and not exited:
                continue
            rank_full = {int(i): rank for rank, i in enumerate(full_top[row], 1)}
            rank_id = {int(i): rank for rank, i in enumerate(id_top[row], 1)}
            for i in sorted(entered | exited):
                i = int(i)
                uvec, ivec = user[int(u)], item[i]
                id_value = float((uvec[:id_dim] * ivec[:id_dim]).sum())
                n_value = float((uvec[id_dim:id_dim + axis_dim]
                                 * ivec[id_dim:id_dim + axis_dim]).sum())
                v_value = float((uvec[id_dim + axis_dim:]
                                 * ivec[id_dim + axis_dim:]).sum())
                full_value = float(full_scores[row, i])
                error = abs(full_value - (id_value + n_value + v_value))
                max_reconstruction_error = max(max_reconstruction_error, error)
                if not np.isclose(full_value, id_value + n_value + v_value,
                                  atol=2e-5, rtol=1e-5):
                    raise ValueError("M5 full score cannot be reconstructed from ID/N/V")
                candidate_rows.append(dict(user=int(u),
                    segment=str(cache.seg[start + row]), item=i,
                    top10_change_from_id=("entered_full" if i in entered else "exited_full"),
                    id_rank=rank_id.get(i), full_rank=rank_full.get(i),
                    id_score=id_value, n_score=n_value, v_score=v_value,
                    nv_score=n_value + v_value, full_score=full_value))
    return result_full, result_id, pd.DataFrame(candidate_rows), max_reconstruction_error


def _metric_rows(prepared: dict, tops: dict[str, np.ndarray]) -> pd.DataFrame:
    v3 = movement.screen.base.v3
    cache, data, meta = prepared["cache"], prepared["data"], prepared["meta"]
    rows = []
    for component, top in tops.items():
        measures = v3.score_topk(top, np.asarray(cache.users), KS,
            cache.pos_key, cache.pos_rev, data["n_items"], cache.P_arr,
            meta["price_pct"], -np.log2(meta["pop_prob"] + 1e-12),
            meta["cat"], cache.ideal)
        for segment in ("전체", "저CLV", "중CLV", "고CLV"):
            subset = np.ones(len(cache.users), dtype=bool) if segment == "전체" else cache.seg == segment
            if not subset.any():
                continue
            for k in KS:
                for metric, values in measures[k].items():
                    rows.append(dict(component=component, segment=segment, cutoff=k,
                        metric=("price_purchase_amount_weighted_hit" if metric == "revenue" else metric),
                        value=float(values[subset].mean()), n_users=int(subset.sum())))
                if segment == "전체":
                    rows.append(dict(component=component, segment=segment, cutoff=k,
                        metric="coverage", value=float(np.unique(top[:, :k]).size / data["n_items"]),
                        n_users=int(subset.sum())))
    return pd.DataFrame(rows)


def _movement_tables(prepared: dict, id_top: np.ndarray, full_top: np.ndarray):
    cache = prepared["cache"]
    truth, users = movement.movement_tables(cache.users, cache.seg, id_top,
        full_top, cache.gt, cache.rev)
    truth = truth.rename(columns={"m4_rank": "id_rank", "m5_rank": "full_rank",
        "m4_bucket": "id_bucket", "m5_bucket": "full_bucket"})
    summary = movement._summary(
        truth.rename(columns={"id_bucket": "m4_bucket", "full_bucket": "m5_bucket"}), users)
    summary = summary.rename(columns={c: c.replace("gained_from", "full_gained_from_id")
        .replace("lost_to", "full_lost_to_id") for c in summary.columns})
    return truth, users, summary


def run(report_path: str | Path, out_dir: str | Path) -> dict:
    """Load exactly one selected development checkpoint; never train or select."""
    report_path, out_dir = Path(report_path), Path(out_dir)
    report, _ = movement._verify_report(report_path)
    cfg, prepared, _ = movement.screen.prepare(report["source_report"], out_dir)
    movement.screen.verified_anchors(report["source_report"], cfg, prepared)
    if (set(prepared["data"]["splits"]) != {"test"}
            or prepared["base_cfg"].get("EVAL_HOLDOUT")
            or prepared["base_cfg"].get("MIN_ITEM_INTER") != 1):
        raise ValueError("Development-only/new-item protocol mismatch")
    movement._verify_new_item_truth(prepared)
    arm = movement._one_arm(report, movement.screen.MODEL_ID)
    movement.screen._selected_arm(arm, cfg, prepared)
    if not movement._same_identity_except_result_location(
            arm.get("identity", {}), movement.screen.identity(prepared, cfg)):
        raise ValueError("Selected M5 identity differs from prepared input/code")
    model = movement._load_model(prepared, cfg, arm, movement.screen.spec())
    full_top, id_top, candidate_scores, score_error = rank_and_explain(model, prepared)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    readback = movement._metric_readback(prepared, full_top, arm)
    tops = dict(zip(COMPONENTS, (full_top, id_top)))
    absolute = _metric_rows(prepared, tops)
    pivot = absolute.pivot(index=["segment", "cutoff", "metric"], columns="component", values="value")
    comparison = pivot.reset_index()
    comparison["full_minus_id"] = (comparison["m5_full"] - comparison["m5_id_component_only"])
    comparison["full_vs_id_pct"] = np.where(comparison["m5_id_component_only"] != 0,
        100 * comparison["full_minus_id"] / comparison["m5_id_component_only"], np.nan)
    truth, users, summary = _movement_tables(prepared, id_top, full_top)
    lookup = comparison.set_index(["segment", "cutoff", "metric"])
    for segment in summary.segment:
        measured = float(summary.set_index("segment").loc[segment, "weighted_net"])
        expected = float(lookup.loc[(segment, 10, "price_purchase_amount_weighted_hit"), "full_minus_id"])
        if not np.isclose(measured, expected, atol=1e-6, rtol=1e-5):
            raise ValueError(f"Top-10 movement does not reconcile in {segment}")
    root = out_dir / "m5_nv_score_component_diagnostic"
    frames = dict(absolute=absolute, comparison=comparison, truth_movements=truth,
                  user_movements=users, candidate_scores=candidate_scores, summary=summary)
    paths = {name: str(root / f"{name}.csv") for name in frames}
    paths["diagnostic"] = str(root / "diagnostic.json")
    save_csv = movement.screen.base.capacity.test10._atomic_csv
    save_json = movement.screen.base.capacity.test10._atomic_json
    for name, frame in frames.items():
        save_csv(Path(paths[name]), frame)
    save_json(Path(paths["diagnostic"]), dict(code_version=VERSION, seed=43,
        split="historical_development_days_684_690", source_result=str(report_path),
        source_result_sha256=movement.SOURCE_RESULT_SHA256,
        selected_checkpoint=dict(path=arm["checkpoint"], sha256=arm["checkpoint_sha256"],
                                 epoch=arm["selected_epoch"]),
        full_score_readback=readback, max_score_reconstruction_error=score_error,
        score_definition="same jointly trained M5; full=ID+N+V, diagnostic ID-only omits N/V only at inference",
        id_component_is_m1=False, id_component_is_deployable_model=False,
        new_training=False, checkpoint_selection_changed=False, final_test=False,
        holdout=False, significance_claim=False, causal_attribution_claim=False,
        paths=paths))
    return paths
