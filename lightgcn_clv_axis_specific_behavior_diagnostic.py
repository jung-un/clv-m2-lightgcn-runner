"""Post-hoc behavior diagnostic for the fixed 10-seed M2 test run.

This module never trains a model and never selects a checkpoint.  It reloads
the already evaluated M1@64 and accepted M2 checkpoints, reconstructs their
Top-50 lists, and explains the fixed test result by train-history N/V user
groups.  The protected test split is used only for descriptive error analysis;
the output must not be used to tune another model on the same test interval.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_joint_nv_model import JointNVLightGCN
from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as final10
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-axis-specific-test10-behavior-diagnostic-v1"
MODELS = ("m1_64", "m2_axis_specific_gate")
GROUP_LABELS = {
    "low_n_low_v": "낮은 거래활동·낮은 거래당 가치",
    "high_n_low_v": "높은 거래활동·낮은 거래당 가치",
    "low_n_high_v": "낮은 거래활동·높은 거래당 가치",
    "high_n_high_v": "높은 거래활동·높은 거래당 가치",
    "invalid_axis": "N/V 산출 불가",
}
ITEM_TRAITS = (
    "repeat_purchase_share",
    "median_repeat_gap",
    "repeat_gap_valid",
    "price_percentile",
    "category_price_percentile",
    "mean_unit_price",
    "mean_transaction_value_share",
    "train_user_count",
)


@dataclass(frozen=True)
class BehaviorDiagnosticConfig:
    test_out_dir: str = ""
    run_json: str = ""
    seeds: tuple[int, ...] = final10.SEEDS
    ks: tuple[int, ...] = (10, 20, 50)
    score_batch_size: int = 64
    top_product_examples: int = 30


def configure_behavior_diagnostic(**overrides) -> BehaviorDiagnosticConfig:
    defaults = {
        "test_out_dir": final10.configure_test10_run().out_dir,
    }
    return validate_behavior_config(
        BehaviorDiagnosticConfig(**(defaults | overrides))
    )


def validate_behavior_config(
    cfg: BehaviorDiagnosticConfig,
) -> BehaviorDiagnosticConfig:
    if tuple(cfg.seeds) != final10.SEEDS:
        raise ValueError("진단은 확정 실행과 같은 seed 42~51을 사용해야 합니다")
    if tuple(cfg.ks) != (10, 20, 50):
        raise ValueError("진단 K는 확정 평가와 같은 10·20·50이어야 합니다")
    if cfg.score_batch_size <= 0 or cfg.top_product_examples <= 0:
        raise ValueError("배치 크기와 예시 상품 수는 양수여야 합니다")
    if not cfg.test_out_dir:
        raise ValueError("10시드 결과 폴더가 필요합니다")
    return cfg


def preflight_summary(cfg: BehaviorDiagnosticConfig) -> dict:
    cfg = validate_behavior_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "source_result": "fixed M2 10-seed final-test checkpoints",
        "models": list(MODELS),
        "seeds": list(cfg.seeds),
        "user_group_source": (
            "train-history q_N and q_V only; test labels never define groups"
        ),
        "groups": GROUP_LABELS,
        "comparison": (
            "new-item test truth versus M1 and M2 recommendations, item traits, "
            "truth-rank transitions, and M2 ID/N/V score contributions"
        ),
        "training": False,
        "checkpoint_selection": False,
        "interpretation_only": True,
        "prohibition": (
            "do not tune a new model on these post-hoc test diagnostics; use a "
            "new predeclared time split or independent data for a changed model"
        ),
        "test_out_dir": cfg.test_out_dir,
    }


def assign_nv_groups(
    q_n: np.ndarray,
    q_v: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Assign four train-history N/V quadrants without looking at test truth."""
    q_n = np.asarray(q_n, dtype=float)
    q_v = np.asarray(q_v, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if q_n.shape != q_v.shape or q_n.shape != valid.shape:
        raise ValueError("q_N, q_V, valid shape이 다릅니다")
    out = np.full(len(valid), "invalid_axis", dtype=object)
    low_n = q_n < 0.5
    low_v = q_v < 0.5
    out[valid & low_n & low_v] = "low_n_low_v"
    out[valid & ~low_n & low_v] = "high_n_low_v"
    out[valid & low_n & ~low_v] = "low_n_high_v"
    out[valid & ~low_n & ~low_v] = "high_n_high_v"
    return out.astype(str)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _find_run_json(cfg: BehaviorDiagnosticConfig) -> tuple[Path, dict, str]:
    root = Path(cfg.test_out_dir)
    candidates = (
        [Path(cfg.run_json)]
        if cfg.run_json
        else sorted(
            root.glob("m2_axis_specific_test10_????????????.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    if not candidates or not candidates[0].exists():
        raise FileNotFoundError(
            "10시드 종합 JSON을 찾지 못했습니다. run_json 또는 test_out_dir를 확인하세요"
        )
    path = candidates[0]
    match = re.fullmatch(r"m2_axis_specific_test10_([0-9a-f]{12})", path.stem)
    if match is None:
        raise ValueError(f"10시드 종합 JSON 파일명이 예상 형식이 아닙니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_schema") != final10.ACCEPTED_M2_FEATURE_SCHEMA:
        raise RuntimeError(
            "선택한 결과의 M2 입력 스키마가 현재 승인된 스키마와 다릅니다"
        )
    return path, payload, match.group(1)


def _arm_paths(root: Path, run_hash: str, model_id: str, seed: int) -> dict:
    stem = root / "arms" / run_hash / f"{model_id}_s{seed}"
    return {
        "result": stem.with_suffix(".json"),
        "checkpoint": stem.with_suffix(".pt"),
    }


def _load_model(
    prepared: dict,
    test_cfg: final10.Test10Config,
    root: Path,
    run_hash: str,
    model_id: str,
    seed: int,
):
    paths = _arm_paths(root, run_hash, model_id, seed)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"완료된 arm 산출물이 없습니다: {path}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if result.get("test_evaluation_count") != 1:
        raise RuntimeError(f"test 평가 횟수가 1이 아닙니다: {paths['result']}")
    if result.get("checkpoint_sha256") != file_sha256(paths["checkpoint"]):
        raise RuntimeError(f"checkpoint hash 불일치: {paths['checkpoint']}")
    model, _ = final10._build_model(prepared, test_cfg, model_id, seed)
    try:
        blob = torch.load(paths["checkpoint"], map_location=v3.DEVICE, weights_only=False)
    except TypeError:  # torch<2.6 compatibility
        blob = torch.load(paths["checkpoint"], map_location=v3.DEVICE)
    if blob.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError(f"현재 데이터와 checkpoint input hash 불일치: {paths['checkpoint']}")
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, result, paths


@torch.no_grad()
def _masked_topk(model, prepared: dict, max_k: int, batch_size: int):
    # M1's legacy evaluator can also construct an unused value block.  The
    # diagnostic needs only the actual ranking embedding, so avoid that extra
    # full-catalog computation while keeping JointNV's compatible interface.
    user_embedding, item_embedding, *_ = model.embeddings(need_value=False)
    users = prepared["cache"].users.astype(np.int64)
    topk = np.empty((len(users), max_k), np.int32)
    csr_ptr = prepared["data"]["csr_ptr"]
    csr_items = prepared["data"]["csr_items"]
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        tensor_users = torch.as_tensor(
            batch_users, dtype=torch.long, device=user_embedding.device
        )
        scores = user_embedding[tensor_users] @ item_embedding.T
        for row, user in enumerate(batch_users):
            left, right = csr_ptr[user], csr_ptr[user + 1]
            if right > left:
                scores[row, csr_items[left:right]] = -1e9
        topk[start : start + len(batch_users)] = (
            scores.topk(max_k, dim=1).indices.cpu().numpy()
        )
    return users, topk, user_embedding, item_embedding


def _hit_matrix(users: np.ndarray, topk: np.ndarray, cache, n_items: int):
    keys = users[:, None].astype(np.int64) * n_items + topk.astype(np.int64)
    positions = np.clip(
        np.searchsorted(cache.pos_key, keys), 0, len(cache.pos_key) - 1
    )
    hits = cache.pos_key[positions] == keys
    gains = np.where(hits, cache.pos_rev[positions], 0.0)
    return hits, gains


def _per_user_metrics(
    users: np.ndarray,
    topk: np.ndarray,
    prepared: dict,
    model_id: str,
    seed: int,
    group_ids: np.ndarray,
) -> pd.DataFrame:
    data = prepared["data"]
    cache = prepared["cache"]
    n_items = data["n_items"]
    meta = prepared["meta"]
    item_novelty = -np.log2(meta["pop_prob"] + 1e-12)
    scored = v3.score_topk(
        topk,
        users,
        [10, 20, 50],
        cache.pos_key,
        cache.pos_rev,
        n_items,
        cache.P_arr,
        meta["price_pct"],
        item_novelty,
        meta["cat"],
        cache.ideal,
    )
    rows = {
        "seed": np.full(len(users), seed),
        "model_id": np.full(len(users), model_id),
        "user_idx": users,
        "group_id": group_ids[users],
        "group_label": [GROUP_LABELS[group_ids[user]] for user in users],
        "truth_item_count": cache.P_arr[users],
        "q_n": prepared["axes"]["q_n"][users],
        "q_v": prepared["axes"]["q_v"][users],
        "repeat_transaction_count": prepared["axes"]["repeat_transaction_count"][users],
        "repeat_transaction_rate": prepared["axes"]["repeat_transaction_rate"][users],
        "mean_transaction_value": prepared["axes"]["mean_transaction_value"][users],
    }
    for k, metrics in scored.items():
        for metric in ("recall", "precision", "ndcg", "hr", "map", "revenue", "arp"):
            public = {
                "revenue": "price_purchase_amount_weighted_hit",
                "arp": "mean_recommended_price_percentile",
            }.get(metric, metric)
            rows[f"{public}@{k}"] = metrics[metric]
    return pd.DataFrame(rows)


def _summarize_user_metrics(per_user: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_user if "@" in column]
    rows = []
    for (seed, group_id), group in per_user.groupby(["seed", "group_id"], sort=False):
        baseline = group[group.model_id.eq("m1_64")].set_index("user_idx")
        model = group[group.model_id.eq("m2_axis_specific_gate")].set_index("user_idx")
        common = baseline.index.intersection(model.index)
        for metric in metric_columns:
            m1 = baseline.loc[common, metric].to_numpy(float)
            m2 = model.loc[common, metric].to_numpy(float)
            delta = m2 - m1
            rows.append(
                {
                    "seed": int(seed),
                    "group_id": group_id,
                    "group_label": GROUP_LABELS[group_id],
                    "n_users": len(common),
                    "metric": metric,
                    "m1_mean": float(m1.mean()),
                    "m2_mean": float(m2.mean()),
                    "mean_delta": float(delta.mean()),
                    "relative_change_pct": (
                        float(delta.mean() / m1.mean() * 100.0)
                        if abs(m1.mean()) > 1e-12
                        else np.nan
                    ),
                    "improved_user_share": float((delta > 0).mean()),
                    "degraded_user_share": float((delta < 0).mean()),
                }
            )
    by_seed = pd.DataFrame(rows)
    mean_rows = []
    for (group_id, metric), group in by_seed.groupby(["group_id", "metric"], sort=False):
        mean_rows.append(
            {
                "group_id": group_id,
                "group_label": GROUP_LABELS[group_id],
                "metric": metric,
                "n_seeds": group.seed.nunique(),
                "mean_n_users": float(group.n_users.mean()),
                "m1_mean": float(group.m1_mean.mean()),
                "m2_mean": float(group.m2_mean.mean()),
                "mean_delta": float(group.mean_delta.mean()),
                "relative_change_pct": (
                    float(group.mean_delta.mean() / group.m1_mean.mean() * 100.0)
                    if abs(group.m1_mean.mean()) > 1e-12
                    else np.nan
                ),
                "improved_user_share": float(group.improved_user_share.mean()),
                "degraded_user_share": float(group.degraded_user_share.mean()),
            }
        )
    return by_seed, pd.DataFrame(mean_rows)


def _raw_item_traits(train: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """Reconstruct human-readable, train-only item traits used by M2."""
    def modal(series):
        mode = series.mode(dropna=True)
        return mode.iat[0] if len(mode) else "UNKNOWN"

    item = train.groupby("i_idx", sort=True).agg(
        item_id=("i_raw", "first"),
        category=("cat_raw", modal),
        train_row_count=("i_idx", "size"),
        train_user_count=("u_idx", "nunique"),
        mean_unit_price=("up", "mean"),
    )
    pairs = train.groupby(["u_idx", "i_idx"], sort=False).size()
    item["repeat_purchase_share"] = pairs.gt(1).groupby(level="i_idx").mean()
    dated = train[["u_idx", "i_idx", "t"]].drop_duplicates().sort_values(
        ["u_idx", "i_idx", "t"]
    )
    gap = dated.groupby(["u_idx", "i_idx"], sort=False)["t"].diff()
    if v3.DCFG["is_date"]:
        gap = gap.dt.total_seconds() / 86400.0
    dated = dated.assign(_gap=np.asarray(gap, dtype=float))
    repeat_gap = dated.groupby("i_idx", sort=False)["_gap"].median()
    item["median_repeat_gap"] = repeat_gap.reindex(item.index)
    item["repeat_gap_valid"] = item.median_repeat_gap.notna().astype(float)
    item["price_percentile"] = item.mean_unit_price.rank(pct=True, method="average")
    item["category_price_percentile"] = item.groupby("category")["mean_unit_price"].rank(
        pct=True, method="average"
    )
    transaction_keys = ["b_raw"] if "b_raw" in train else ["u_idx", "t"]
    item_in_transaction = train.groupby([*transaction_keys, "i_idx"], sort=False).v.sum()
    levels = list(range(len(transaction_keys)))
    total = item_in_transaction.groupby(level=levels, sort=False).transform("sum")
    share = item_in_transaction.div(total.where(total > 0)).fillna(0.0)
    item["mean_transaction_value_share"] = share.groupby(level="i_idx").mean()
    item.index.name = "item_idx"
    result = pd.DataFrame({"item_idx": np.arange(n_items, dtype=int)})
    result = result.merge(item.reset_index(), on="item_idx", how="left")
    return result


def _role_items(
    user: int,
    truth: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> dict[str, np.ndarray]:
    truth_set = set(map(int, truth))
    m1_set = set(map(int, m1[:10]))
    m2_set = set(map(int, m2[:10]))
    promoted = np.array([item for item in m2[:10] if int(item) not in m1_set], dtype=int)
    displaced = np.array([item for item in m1[:10] if int(item) not in m2_set], dtype=int)
    return {
        "test_truth_new_items": np.asarray(truth, dtype=int),
        "m1_top10": np.asarray(m1[:10], dtype=int),
        "m2_top10": np.asarray(m2[:10], dtype=int),
        "m1_hit_top10": np.array([item for item in m1[:10] if int(item) in truth_set], dtype=int),
        "m2_hit_top10": np.array([item for item in m2[:10] if int(item) in truth_set], dtype=int),
        "m2_promoted_top10": promoted,
        "m2_promoted_hit": np.array([item for item in promoted if int(item) in truth_set], dtype=int),
        "m2_promoted_miss": np.array([item for item in promoted if int(item) not in truth_set], dtype=int),
        "m1_displaced_top10": displaced,
        "m1_displaced_hit": np.array([item for item in displaced if int(item) in truth_set], dtype=int),
    }


def _pair_components(
    model: JointNVLightGCN,
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    user: int,
    items: np.ndarray,
) -> dict[str, float]:
    if not len(items):
        return {
            "score_id": np.nan,
            "score_n": np.nan,
            "score_v": np.nan,
            "score_total": np.nan,
            "nv_absolute_share": np.nan,
        }
    item_t = torch.as_tensor(items, dtype=torch.long, device=item_embedding.device)
    u = user_embedding[user]
    selected = item_embedding[item_t]
    id_end = model.id_dim
    n_end = id_end + model.axis_dim
    score_id = (selected[:, :id_end] * u[:id_end]).sum(1)
    score_n = (selected[:, id_end:n_end] * u[id_end:n_end]).sum(1)
    score_v = (selected[:, n_end:] * u[n_end:]).sum(1)
    total = score_id + score_n + score_v
    denominator = score_id.abs() + score_n.abs() + score_v.abs() + 1e-12
    return {
        "score_id": float(score_id.mean()),
        "score_n": float(score_n.mean()),
        "score_v": float(score_v.mean()),
        "score_total": float(total.mean()),
        "nv_absolute_share": float(((score_n.abs() + score_v.abs()) / denominator).mean()),
        "decomposition_max_abs_error": float(
            (total - (selected * u).sum(1)).abs().max()
        ),
    }


def _item_role_records(
    seed: int,
    users: np.ndarray,
    m1_topk: np.ndarray,
    m2_topk: np.ndarray,
    group_ids: np.ndarray,
    prepared: dict,
    item_traits: pd.DataFrame,
    m2_model: JointNVLightGCN,
    m2_user_embedding: torch.Tensor,
    m2_item_embedding: torch.Tensor,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    traits = item_traits.set_index("item_idx")
    records = []
    promoted_occurrences = []
    transition_occurrences = []
    gt = prepared["cache"].gt
    rev = prepared["cache"].rev
    for row, user in enumerate(users):
        truth = np.asarray(gt[int(user)], dtype=int)
        roles = _role_items(int(user), truth, m1_topk[row], m2_topk[row])
        truth_amount = {int(item): float(amount) for item, amount in zip(truth, rev[int(user)])}
        for role, items in roles.items():
            selected = traits.reindex(items)
            record = {
                "seed": seed,
                "user_idx": int(user),
                "group_id": group_ids[user],
                "group_label": GROUP_LABELS[group_ids[user]],
                "role": role,
                "n_items": len(items),
                "n_unique_items": len(np.unique(items)),
                "mean_test_purchase_amount": (
                    float(np.mean([truth_amount.get(int(item), 0.0) for item in items]))
                    if len(items)
                    else np.nan
                ),
            }
            for trait in ITEM_TRAITS:
                record[trait] = float(selected[trait].mean()) if len(items) else np.nan
            if role.startswith("m2_") or role == "test_truth_new_items":
                record.update(
                    _pair_components(
                        m2_model,
                        m2_user_embedding,
                        m2_item_embedding,
                        int(user),
                        items,
                    )
                )
            records.append(record)
        for item in roles["m2_promoted_top10"]:
            promoted_occurrences.append(
                {
                    "seed": seed,
                    "user_idx": int(user),
                    "group_id": group_ids[user],
                    "item_idx": int(item),
                    "hit": int(item) in truth_amount,
                    "test_purchase_amount": truth_amount.get(int(item), 0.0),
                    "m2_rank": int(np.where(m2_topk[row, :10] == item)[0][0] + 1),
                }
            )
        for transition, items in (
            ("gained_hit", roles["m2_promoted_hit"]),
            ("lost_hit", roles["m1_displaced_hit"]),
        ):
            for item in items:
                transition_occurrences.append(
                    {
                        "seed": seed,
                        "user_idx": int(user),
                        "group_id": group_ids[user],
                        "transition": transition,
                        "item_idx": int(item),
                        "test_purchase_amount": truth_amount[int(item)],
                    }
                )
    return pd.DataFrame(records), promoted_occurrences, transition_occurrences


def _truth_rank_transitions(
    seed: int,
    users: np.ndarray,
    m1_topk: np.ndarray,
    m2_topk: np.ndarray,
    group_ids: np.ndarray,
    prepared: dict,
    item_traits: pd.DataFrame,
) -> pd.DataFrame:
    traits = item_traits.set_index("item_idx")
    rows = []
    for row, user in enumerate(users):
        m1_rank = {int(item): rank + 1 for rank, item in enumerate(m1_topk[row])}
        m2_rank = {int(item): rank + 1 for rank, item in enumerate(m2_topk[row])}
        truth = prepared["cache"].gt[int(user)]
        amount = prepared["cache"].rev[int(user)]
        for item, value in zip(truth, amount):
            item = int(item)
            r1 = m1_rank.get(item, 51)
            r2 = m2_rank.get(item, 51)
            record = {
                "seed": seed,
                "user_idx": int(user),
                "group_id": group_ids[user],
                "group_label": GROUP_LABELS[group_ids[user]],
                "item_idx": item,
                "test_purchase_amount": float(value),
                "m1_rank_capped_51": r1,
                "m2_rank_capped_51": r2,
                "rank_improvement": r1 - r2,
                "entered_top10": r1 > 10 and r2 <= 10,
                "left_top10": r1 <= 10 and r2 > 10,
                "entered_top20": r1 > 20 and r2 <= 20,
                "left_top20": r1 <= 20 and r2 > 20,
            }
            item_row = traits.loc[item]
            record.update({trait: item_row[trait] for trait in ITEM_TRAITS})
            rows.append(record)
    return pd.DataFrame(rows)


def _aggregate_item_roles(per_user_roles: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n_items",
        "n_unique_items",
        "mean_test_purchase_amount",
        *ITEM_TRAITS,
        "score_id",
        "score_n",
        "score_v",
        "score_total",
        "nv_absolute_share",
    ]
    available = [column for column in numeric if column in per_user_roles]
    return (
        per_user_roles.groupby(["seed", "group_id", "group_label", "role"], dropna=False)[available]
        .mean()
        .reset_index()
    )


def _product_examples(
    occurrences: list[dict],
    item_traits: pd.DataFrame,
    cfg: BehaviorDiagnosticConfig,
) -> pd.DataFrame:
    if not occurrences:
        return pd.DataFrame()
    frame = pd.DataFrame(occurrences)
    frame["seed_user"] = frame.seed.astype(str) + ":" + frame.user_idx.astype(str)
    grouped = frame.groupby(["group_id", "item_idx"], sort=False).agg(
        promotion_count=("item_idx", "size"),
        promotion_seed_user_count=("seed_user", "nunique"),
        hit_count=("hit", "sum"),
        mean_m2_rank=("m2_rank", "mean"),
        mean_test_purchase_amount=("test_purchase_amount", "mean"),
    ).reset_index()
    grouped["hit_rate_when_promoted"] = grouped.hit_count / grouped.promotion_count
    grouped["group_label"] = grouped.group_id.map(GROUP_LABELS)
    grouped = grouped.merge(item_traits, on="item_idx", how="left")
    return (
        grouped.sort_values(
            ["group_id", "promotion_count", "hit_count"],
            ascending=[True, False, False],
        )
        .groupby("group_id", sort=False)
        .head(cfg.top_product_examples)
        .reset_index(drop=True)
    )


def _transition_examples(
    occurrences: list[dict], item_traits: pd.DataFrame, cfg: BehaviorDiagnosticConfig
) -> pd.DataFrame:
    if not occurrences:
        return pd.DataFrame()
    frame = pd.DataFrame(occurrences)
    frame["seed_user"] = frame.seed.astype(str) + ":" + frame.user_idx.astype(str)
    grouped = frame.groupby(["group_id", "transition", "item_idx"], sort=False).agg(
        occurrence_count=("item_idx", "size"),
        seed_user_count=("seed_user", "nunique"),
        mean_test_purchase_amount=("test_purchase_amount", "mean"),
    ).reset_index()
    grouped["group_label"] = grouped.group_id.map(GROUP_LABELS)
    grouped = grouped.merge(item_traits, on="item_idx", how="left")
    return (
        grouped.sort_values(
            ["group_id", "transition", "occurrence_count"],
            ascending=[True, True, False],
        )
        .groupby(["group_id", "transition"], sort=False)
        .head(cfg.top_product_examples)
        .reset_index(drop=True)
    )


def _mean_across_seeds(
    frame: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Average already-computed seed summaries without pooling seed rows."""
    if frame.empty:
        return frame.copy()
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    return (
        frame.groupby(keys, dropna=False, sort=False)[numeric]
        .mean()
        .reset_index()
        .assign(n_seeds=frame.seed.nunique())
    )


def _truth_rank_summary(truth_rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (seed, group_id), group in truth_rank.groupby(
        ["seed", "group_id"], sort=False
    ):
        rows.append(
            {
                "seed": int(seed),
                "group_id": group_id,
                "group_label": GROUP_LABELS[group_id],
                "n_truth_items": len(group),
                "mean_m1_rank_capped_51": float(group.m1_rank_capped_51.mean()),
                "mean_m2_rank_capped_51": float(group.m2_rank_capped_51.mean()),
                "mean_rank_improvement": float(group.rank_improvement.mean()),
                "entered_top10_share": float(group.entered_top10.mean()),
                "left_top10_share": float(group.left_top10.mean()),
                "entered_top20_share": float(group.entered_top20.mean()),
                "left_top20_share": float(group.left_top20.mean()),
            }
        )
    by_seed = pd.DataFrame(rows)
    return by_seed, _mean_across_seeds(
        by_seed, ["group_id", "group_label"]
    )


def _quality_rows(
    seed: int,
    model_id: str,
    users: np.ndarray,
    topk: np.ndarray,
    per_user: pd.DataFrame,
    prepared: dict,
    result: dict,
    decomposition_error: float | None,
) -> list[dict]:
    rows = []
    n_items = prepared["data"]["n_items"]
    train_key = prepared["data"]["pos_key"]
    top_key = users[:, None].astype(np.int64) * n_items + topk.astype(np.int64)
    position = np.clip(np.searchsorted(train_key, top_key), 0, len(train_key) - 1)
    train_overlap = int((train_key[position] == top_key).sum())
    rows.append({"seed": seed, "model_id": model_id, "check": "topk_excludes_train_pairs", "value": train_overlap, "passed": train_overlap == 0})
    rows.append({"seed": seed, "model_id": model_id, "check": "topk_has_no_duplicates", "value": int(sum(len(set(row)) != len(row) for row in topk)), "passed": all(len(set(row)) == len(row) for row in topk)})
    for metric in ("recall@10", "ndcg@10", "price_purchase_amount_weighted_hit@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"):
        observed = float(per_user[metric].mean())
        expected = float(result["metrics"][metric])
        error = abs(observed - expected)
        rows.append({"seed": seed, "model_id": model_id, "check": f"recomputed_{metric}", "value": error, "passed": error < 1e-7})
    if decomposition_error is not None:
        rows.append({"seed": seed, "model_id": model_id, "check": "id_n_v_score_decomposition", "value": decomposition_error, "passed": decomposition_error < 1e-5})
    return rows


def _persist(
    cfg: BehaviorDiagnosticConfig,
    run_hash: str,
    source_run_json: Path,
    prepared: dict,
    per_user: pd.DataFrame,
    group_by_seed: pd.DataFrame,
    group_mean: pd.DataFrame,
    item_role_per_user: pd.DataFrame,
    item_role_summary: pd.DataFrame,
    item_role_mean: pd.DataFrame,
    truth_rank: pd.DataFrame,
    truth_rank_by_seed: pd.DataFrame,
    truth_rank_mean: pd.DataFrame,
    promoted_examples: pd.DataFrame,
    transition_examples: pd.DataFrame,
    quality: pd.DataFrame,
    source_paths: list[dict],
) -> dict:
    payload_hash = hashlib.sha256(
        json.dumps(
            {"version": CODE_VERSION, "run_hash": run_hash, "config": asdict(cfg)},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:12]
    root = Path(cfg.test_out_dir) / "behavior_diagnostic" / payload_hash
    paths = {
        "user_metric_by_seed": root / "user_metric_by_seed.csv",
        "user_group_by_seed": root / "user_group_summary_by_seed.csv",
        "user_group_10seed_mean": root / "user_group_summary_10seed_mean.csv",
        "item_role_by_user": root / "item_role_by_user.csv",
        "item_role_summary": root / "item_role_summary_by_seed.csv",
        "item_role_10seed_mean": root / "item_role_summary_10seed_mean.csv",
        "truth_rank_transition": root / "truth_rank_transition.csv",
        "truth_rank_by_seed": root / "truth_rank_summary_by_seed.csv",
        "truth_rank_10seed_mean": root / "truth_rank_summary_10seed_mean.csv",
        "promoted_product_examples": root / "promoted_product_examples.csv",
        "truth_hit_transition_examples": root / "truth_hit_transition_examples.csv",
        "quality_checks": root / "quality_checks.csv",
        "json": root / "diagnostic.json",
    }
    frames = {
        "user_metric_by_seed": per_user,
        "user_group_by_seed": group_by_seed,
        "user_group_10seed_mean": group_mean,
        "item_role_by_user": item_role_per_user,
        "item_role_summary": item_role_summary,
        "item_role_10seed_mean": item_role_mean,
        "truth_rank_transition": truth_rank,
        "truth_rank_by_seed": truth_rank_by_seed,
        "truth_rank_10seed_mean": truth_rank_mean,
        "promoted_product_examples": promoted_examples,
        "truth_hit_transition_examples": transition_examples,
        "quality_checks": quality,
    }
    for name, frame in frames.items():
        _atomic_csv(paths[name], frame)
    payload = {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc error analysis of fixed final test checkpoints",
        "source_run_json": str(source_run_json),
        "source_run_hash": run_hash,
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "user_group_definition": {
            "source": "train-history fixed q_N/q_V percentiles",
            "threshold": 0.5,
            "labels": GROUP_LABELS,
        },
        "source_checkpoints": source_paths,
        "quality_passed": bool(len(quality) and quality.passed.all()),
        "quality_failed": quality.loc[~quality.passed].to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "interpretation_limits": [
            "The test split did not define user groups or select checkpoints.",
            "The diagnostic explains the fixed test result and is not new confirmatory evidence.",
            "A changed model must be developed on a new predeclared time split or independent dataset, not tuned on this test diagnostic.",
            "price/purchase-amount weighted hit is not actual incremental revenue.",
        ],
    }
    _atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_behavior_diagnostic(
    cfg: BehaviorDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_behavior_config(cfg or configure_behavior_diagnostic())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    run_json, run_payload, run_hash = _find_run_json(cfg)
    root = Path(cfg.test_out_dir)
    if run_payload.get("code_version") != final10.CODE_VERSION:
        raise RuntimeError(
            "진단 코드와 10시드 결과 runner 버전이 다릅니다: "
            f"result={run_payload.get('code_version')}, current={final10.CODE_VERSION}"
        )
    stored_config = dict(run_payload["config"])
    stored_config["out_dir"] = str(root)
    stored_config["seeds"] = tuple(stored_config["seeds"])
    test_cfg = final10.validate_test10_config(final10.Test10Config(**stored_config))
    prepared = final10._prepare(test_cfg)
    stored_manifest_hash = final10.moe.manifest_hash(run_payload["input_manifest"])
    if stored_manifest_hash != prepared["input_hash"]:
        raise RuntimeError("10시드 결과와 현재 원천 데이터 manifest가 다릅니다")

    group_ids = assign_nv_groups(
        prepared["axes"]["q_n"],
        prepared["axes"]["q_v"],
        prepared["axes"]["valid_user"],
    )
    item_traits = _raw_item_traits(
        prepared["data"]["train"], prepared["data"]["n_items"]
    )
    all_per_user = []
    all_roles = []
    all_truth_rank = []
    all_promoted = []
    all_transitions = []
    quality_rows = []
    source_paths = []

    for seed in cfg.seeds:
        loaded = {}
        for model_id in MODELS:
            print(f"\n===== 진단 seed {seed} | {model_id} =====")
            model, result, paths = _load_model(
                prepared, test_cfg, root, run_hash, model_id, seed
            )
            users, topk, user_embedding, item_embedding = _masked_topk(
                model, prepared, max(cfg.ks), cfg.score_batch_size
            )
            frame = _per_user_metrics(
                users, topk, prepared, model_id, seed, group_ids
            )
            all_per_user.append(frame)
            loaded[model_id] = {
                "model": model,
                "result": result,
                "paths": paths,
                "users": users,
                "topk": topk,
                "user_embedding": user_embedding,
                "item_embedding": item_embedding,
                "per_user": frame,
            }
            source_paths.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "result": str(paths["result"]),
                    "checkpoint": str(paths["checkpoint"]),
                    "checkpoint_sha256": file_sha256(paths["checkpoint"]),
                }
            )

        m1 = loaded["m1_64"]
        m2 = loaded["m2_axis_specific_gate"]
        if not np.array_equal(m1["users"], m2["users"]):
            raise RuntimeError(f"seed {seed}: M1과 M2 평가사용자 순서가 다릅니다")
        roles, promoted, transitions = _item_role_records(
            seed,
            m1["users"],
            m1["topk"],
            m2["topk"],
            group_ids,
            prepared,
            item_traits,
            m2["model"],
            m2["user_embedding"],
            m2["item_embedding"],
        )
        all_roles.append(roles)
        all_promoted.extend(promoted)
        all_transitions.extend(transitions)
        all_truth_rank.append(
            _truth_rank_transitions(
                seed,
                m1["users"],
                m1["topk"],
                m2["topk"],
                group_ids,
                prepared,
                item_traits,
            )
        )
        decomposition_error = float(
            roles.decomposition_max_abs_error.dropna().max()
        ) if roles.decomposition_max_abs_error.notna().any() else np.nan
        for model_id in MODELS:
            entry = loaded[model_id]
            quality_rows.extend(
                _quality_rows(
                    seed,
                    model_id,
                    entry["users"],
                    entry["topk"],
                    entry["per_user"],
                    prepared,
                    entry["result"],
                    decomposition_error if model_id == "m2_axis_specific_gate" else None,
                )
            )

    per_user = pd.concat(all_per_user, ignore_index=True)
    group_by_seed, group_mean = _summarize_user_metrics(per_user)
    item_role_per_user = pd.concat(all_roles, ignore_index=True)
    item_role_summary = _aggregate_item_roles(item_role_per_user)
    item_role_mean = _mean_across_seeds(
        item_role_summary, ["group_id", "group_label", "role"]
    )
    truth_rank = pd.concat(all_truth_rank, ignore_index=True)
    truth_rank_by_seed, truth_rank_mean = _truth_rank_summary(truth_rank)
    promoted_examples = _product_examples(all_promoted, item_traits, cfg)
    transition_examples = _transition_examples(
        all_transitions, item_traits, cfg
    )
    quality = pd.DataFrame(quality_rows)
    paths = _persist(
        cfg,
        run_hash,
        run_json,
        prepared,
        per_user,
        group_by_seed,
        group_mean,
        item_role_per_user,
        item_role_summary,
        item_role_mean,
        truth_rank,
        truth_rank_by_seed,
        truth_rank_mean,
        promoted_examples,
        transition_examples,
        quality,
        source_paths,
    )
    group_mean.attrs["result_paths"] = paths
    print("\n===== N/V 사용자집단별 10시드 평균 =====")
    focus = group_mean[
        group_mean.metric.isin(
            [
                "recall@10",
                "recall@20",
                "recall@50",
                "ndcg@10",
                "map@10",
                "price_purchase_amount_weighted_hit@10",
            ]
        )
    ]
    print(focus.to_string(index=False))
    print("\n결과 파일:", paths)
    return group_mean


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_behavior_diagnostic()), ensure_ascii=False, indent=2))
