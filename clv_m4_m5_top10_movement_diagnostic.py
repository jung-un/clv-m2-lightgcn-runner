"""Read-only Top-10 movement audit for the selected seed-43 M4/M5 checkpoints.

This is a development-split diagnostic, not a checkpoint selection or a new
recommendation model. The matched M1 has aggregate metrics but no saved
selected weights, so user-level M1 movements are deliberately unavailable.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m5_linear_nv_original_m4_lambda025_screen as screen
from clv_run_state import file_sha256


VERSION = "m4-m5-selected-top10-movement-development-v1"
SOURCE_RESULT_SHA256 = "bdc482b28baeea0e0792f6da10d9cd29d920e5018694c68c024f706cd7d16556"
MODEL_IDS = (screen.lambda025.MODEL_ID, screen.MODEL_ID)
TOP_K = 50


def _one_arm(report: dict, model_id: str) -> dict:
    matches = [arm for arm in report.get("arms", [])
               if arm.get("model_id") == model_id and arm.get("seed") == 43]
    if len(matches) != 1:
        raise ValueError(f"Exactly one seed-43 {model_id} is required")
    return matches[0]


def _verify_report(path: Path) -> tuple[dict, object]:
    if not path.is_file() or file_sha256(path) != SOURCE_RESULT_SHA256:
        raise ValueError("Exact completed M5 report is missing/changed; no model loaded")
    report = json.loads(path.read_text(encoding="utf-8"))
    source_cfg = screen.configure(report["config"]["out_dir"])
    expected = json.loads(json.dumps(asdict(source_cfg)))
    actual = report["config"]
    if (report.get("code_version") != screen.VERSION
            or report.get("final_test") is not False
            or report.get("holdout") is not False
            or {k: v for k, v in actual.items() if k != "reuse_dirs"}
            != {k: v for k, v in expected.items() if k != "reuse_dirs"}
            or [Path(p).name for p in actual.get("reuse_dirs", [])]
            != [Path(p).name for p in expected["reuse_dirs"]]
            or report.get("source_report_sha256") != screen.SOURCE_REPORT_SHA):
        raise ValueError("Source report protocol/configuration mismatch")
    if _one_arm(report, "m1").get("checkpoint") is not None:
        raise ValueError("This exact report unexpectedly has M1 weights; review diagnostic scope")
    for model_id in MODEL_IDS:
        screen._selected_arm(_one_arm(report, model_id), source_cfg)
    return report, source_cfg


def _load_model(prepared: dict, cfg, arm: dict, spec: dict):
    checkpoint = Path(arm["checkpoint"])
    if not checkpoint.is_file() or file_sha256(checkpoint) != arm["checkpoint_sha256"]:
        raise ValueError(f"Selected checkpoint missing/changed: {arm['model_id']}")
    if arm.get("identity", {}).get("input_hash") != prepared["input_hash"]:
        raise ValueError(f"Input hash mismatch: {arm['model_id']}")
    model = screen.es.fixed._build(prepared, screen.es.strength_cfg(cfg), spec, 43)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("epoch") != arm["selected_epoch"] or "model_state" not in state:
        raise ValueError(f"Selected checkpoint epoch/state mismatch: {arm['model_id']}")
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _top50(model, prepared: dict) -> np.ndarray:
    v3 = screen.base.v3
    model.eval()
    up, ip, uv, iv = model.embeddings()
    data, cache = prepared["data"], prepared["cache"]
    users = np.asarray(cache.users, dtype=np.int64)
    result = np.empty((len(users), TOP_K), dtype=np.int64)
    ones = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    batch_size = min(int(prepared["base_cfg"]["EVAL_BATCH"]), 256)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        user_tensor = torch.as_tensor(batch, dtype=torch.long, device=v3.DEVICE)
        scores = v3.combined_score_all(up, ip, uv, iv, ones, 0.0, user_tensor)
        for row, user in enumerate(batch):
            begin, end = data["csr_ptr"][user], data["csr_ptr"][user + 1]
            if end > begin:
                scores[row, data["csr_items"][begin:end]] = -1e9
        result[start:start + len(batch)] = scores.topk(TOP_K, dim=1).indices.cpu().numpy()
    return result


def _bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def movement_tables(users, segments, m4_top, m5_top, truth, values):
    """Describe every held-out truth item's rank transition and Top-10 gain/loss."""
    users = np.asarray(users, dtype=np.int64)
    segments = np.asarray(segments)
    if (m4_top.shape != (len(users), TOP_K) or m5_top.shape != m4_top.shape
            or segments.shape != users.shape):
        raise ValueError("User/segment/Top-50 shapes differ")
    if len(set(users.tolist())) != len(users):
        raise ValueError("Evaluation users are duplicated")
    item_rows, user_rows = [], []
    for row, user in enumerate(users):
        user = int(user)
        actual = np.asarray(truth[user], dtype=np.int64)
        weight = np.asarray(values[user], dtype=float)
        if len(actual) != len(weight) or len(actual) == 0 or not np.isfinite(weight).all():
            raise ValueError(f"Invalid held-out truth/value rows for user {user}")
        if len(set(actual.tolist())) != len(actual):
            raise ValueError(f"Duplicate held-out truth items for user {user}")
        rank4 = {int(item): idx + 1 for idx, item in enumerate(m4_top[row])}
        rank5 = {int(item): idx + 1 for idx, item in enumerate(m5_top[row])}
        if len(rank4) != TOP_K or len(rank5) != TOP_K:
            raise ValueError(f"Duplicate candidate in Top-50 for user {user}")
        entered = set(m5_top[row, :10]) - set(m4_top[row, :10])
        exited = set(m4_top[row, :10]) - set(m5_top[row, :10])
        gain_count = loss_count = 0
        gain_weight = loss_weight = 0.0
        for item, value in zip(actual, weight):
            item = int(item)
            r4, r5 = rank4.get(item, 51), rank5.get(item, 51)
            gained = r4 > 10 and r5 <= 10
            lost = r4 <= 10 and r5 > 10
            gain_count += int(gained)
            loss_count += int(lost)
            gain_weight += float(value) if gained else 0.0
            loss_weight += float(value) if lost else 0.0
            item_rows.append(dict(user=user, segment=str(segments[row]), item=item,
                truth_weight=float(value), m4_rank=r4 if r4 <= 50 else None,
                m5_rank=r5 if r5 <= 50 else None, m4_bucket=_bucket(r4),
                m5_bucket=_bucket(r5), top10_change=("gained" if gained else
                "lost" if lost else "retained" if r4 <= 10 else "outside")))
        user_rows.append(dict(user=user, segment=str(segments[row]), truth_count=len(actual),
            top10_candidate_entries=len(entered), top10_candidate_exits=len(exited),
            top10_truth_gained=gain_count, top10_truth_lost=loss_count,
            top10_weight_gained=gain_weight, top10_weight_lost=loss_weight,
            top10_weight_net=gain_weight-loss_weight,
            top10_recall_net=(gain_count-loss_count)/len(actual)))
    return pd.DataFrame(item_rows), pd.DataFrame(user_rows)


def _verify_new_item_truth(prepared):
    cache, data = prepared["cache"], prepared["data"]
    for user, items in cache.gt.items():
        begin, end = data["csr_ptr"][user], data["csr_ptr"][user + 1]
        if np.isin(items, data["csr_items"][begin:end]).any():
            raise ValueError("Train pair appears in held-out truth")


def _metric_readback(prepared, top, arm):
    v3, cache, data = screen.base.v3, prepared["cache"], prepared["data"]
    meta = prepared["meta"]
    measures = v3.score_topk(top, np.asarray(cache.users), (10, 20, 50),
        cache.pos_key, cache.pos_rev, data["n_items"], cache.P_arr,
        meta["price_pct"], -np.log2(meta["pop_prob"] + 1e-12), meta["cat"], cache.ideal)
    checks = {}
    for cutoff in (10, 20, 50):
        for metric, name in (("recall", "recall"), ("ndcg", "ndcg"),
                             ("revenue", "price_purchase_amount_weighted_hit"),
                             ("vndcg", "vndcg")):
            key = f"{name}@{cutoff}"
            measured = float(measures[cutoff][metric].mean())
            expected = float(arm["metrics"][key])
            checks[key] = dict(measured=measured, expected=expected,
                               absolute_difference=abs(measured-expected))
            if not np.isclose(measured, expected, atol=1e-6, rtol=1e-5):
                raise ValueError(f"Selected metric readback mismatch: {arm['model_id']} {key}")
    return checks


def _summary(items, per_user):
    rows = []
    for segment in ("전체", "저CLV", "중CLV", "고CLV"):
        u = per_user if segment == "전체" else per_user[per_user.segment == segment]
        i = items if segment == "전체" else items[items.segment == segment]
        if u.empty:
            continue
        gain = i[i.top10_change == "gained"]
        loss = i[i.top10_change == "lost"]
        row = dict(segment=segment, n_users=len(u), n_truth=len(i),
            truth_gained=len(gain), truth_lost=len(loss),
            weighted_gained=float(gain.truth_weight.sum())/len(u),
            weighted_lost=float(loss.truth_weight.sum())/len(u),
            weighted_net=float(u.top10_weight_net.mean()),
            recall_net=float(u.top10_recall_net.mean()),
            mean_top10_candidate_entries=float(u.top10_candidate_entries.mean()))
        for label, frame in (("gained_from", gain), ("lost_to", loss)):
            col = "m4_bucket" if label == "gained_from" else "m5_bucket"
            for bucket in ("11-20", "21-50", ">50"):
                sub = frame[frame[col] == bucket]
                row[f"{label}_{bucket}_count"] = len(sub)
                row[f"{label}_{bucket}_weighted_per_user"] = float(sub.truth_weight.sum())/len(u)
        rows.append(row)
    return pd.DataFrame(rows)


def run(report_path: str | Path, out_dir: str | Path) -> dict:
    """No optimizer, training loop, M1 refit, test, or holdout construction."""
    report_path, out_dir = Path(report_path), Path(out_dir)
    report, _ = _verify_report(report_path)
    cfg, prepared, _ = screen.prepare(report["source_report"], out_dir)
    screen.verified_anchors(report["source_report"], cfg, prepared)
    if (prepared["data"]["splits"].keys() != {"test"}
            or prepared["base_cfg"].get("EVAL_HOLDOUT")
            or prepared["base_cfg"].get("MIN_ITEM_INTER") != 1):
        raise ValueError("Development-only/new-item protocol mismatch")
    _verify_new_item_truth(prepared)
    arms = {model_id: _one_arm(report, model_id) for model_id in MODEL_IDS}
    if arms[screen.MODEL_ID].get("identity") != screen.identity(prepared, cfg):
        raise ValueError("M5 source identity differs from prepared input/code")
    tops, readbacks = {}, {}
    for model_id, spec in ((screen.lambda025.MODEL_ID, screen.lambda025.spec()),
                           (screen.MODEL_ID, screen.spec())):
        arm = arms[model_id]
        screen._selected_arm(arm, cfg, prepared)
        model = _load_model(prepared, cfg, arm, spec)
        top = _top50(model, prepared)
        readbacks[model_id] = _metric_readback(prepared, top, arm)
        tops[model_id] = top
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache = prepared["cache"]
    items, users = movement_tables(cache.users, cache.seg,
        tops[screen.lambda025.MODEL_ID], tops[screen.MODEL_ID], cache.gt, cache.rev)
    summary = _summary(items, users)
    overall = summary.set_index("segment").loc["전체"]
    reported_delta = (arms[screen.MODEL_ID]["metrics"]["price_purchase_amount_weighted_hit@10"]
                      - arms[screen.lambda025.MODEL_ID]["metrics"]["price_purchase_amount_weighted_hit@10"])
    if not np.isclose(overall.weighted_net, reported_delta, atol=1e-6, rtol=1e-5):
        raise ValueError("Top-10 truth movements do not reconcile to reported weighted-hit delta")
    root = out_dir / "top10_movement_diagnostic"
    paths = {name: str(root / f"{name}.csv") for name in ("truth_movements", "user_movements", "summary")}
    paths["diagnostic"] = str(root / "diagnostic.json")
    save_csv = screen.base.capacity.test10._atomic_csv
    save_json = screen.base.capacity.test10._atomic_json
    for name, frame in (("truth_movements", items), ("user_movements", users), ("summary", summary)):
        save_csv(Path(paths[name]), frame)
    save_json(Path(paths["diagnostic"]), dict(code_version=VERSION, seed=43,
        split="historical_development_days_684_690", source_result=str(report_path),
        source_result_sha256=SOURCE_RESULT_SHA256, selected_checkpoints={mid: dict(
            path=arms[mid]["checkpoint"], sha256=arms[mid]["checkpoint_sha256"],
            epoch=arms[mid]["selected_epoch"]) for mid in MODEL_IDS},
        metric_readback=readbacks, reported_weighted_hit_delta=reported_delta,
        movement_weighted_hit_delta=float(overall.weighted_net),
        m1_user_level_available=False,
        m1_selected_epoch=_one_arm(report, "m1")["selected_epoch"],
        m1_aggregate_metrics={key: _one_arm(report, "m1")["metrics"][key] for key in (
            "recall@10", "ndcg@10", "price_purchase_amount_weighted_hit@10", "vndcg@10")},
        m1_note="Matched seed-43 M1 selected checkpoint is null; aggregate metrics only. No M1 refit or ID-only substitution.",
        exploratory_only=True, selection_changed=False, new_training=False,
        final_test=False, holdout=False, significance_claim=False, paths=paths))
    return paths
